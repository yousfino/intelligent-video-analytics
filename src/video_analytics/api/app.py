
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from video_analytics.api.state import worker_state
from video_analytics.api.worker import AnalyticsWorker

from fastapi import Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from video_analytics.api.schemas import EventListResponse
from video_analytics.database.repository import (
    create_event_repository,
)

from fastapi.responses import Response

from video_analytics.api.prometheus_metrics import (
    render_metrics,
    CONTENT_TYPE_LATEST,
)

from fastapi.responses import JSONResponse

from video_analytics.api.readiness import (
    evaluate_readiness,
)

from video_analytics.api.watchdog import WorkerWatchdog

from fastapi import Request

from video_analytics.database.health import (
    evaluate_writer_health,
)

from video_analytics.api.prometheus_metrics import (
    update_database_metrics,
    update_database_health_metrics,
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def parse_source(value: str) -> str | int:
    return int(value) if value.isdecimal() else value



@asynccontextmanager
async def lifespan(app: FastAPI):
    source = parse_source(
        os.getenv(
            "ANALYTICS_SOURCE",
            "inputs/test_video.mp4",
        )
    )

    model = os.getenv(
        "ANALYTICS_MODEL",
        "yolo",
    )

    loop_file = (
        os.getenv(
            "ANALYTICS_LOOP",
            "false",
        ).lower() == "true"
    )

    worker = AnalyticsWorker(
        source=source,
        model=model,
        loop_file=loop_file,
        config_path=os.getenv(
            "ANALYTICS_CONFIG",
            "configs/pipeline.yaml",
        ),
    )

    watchdog_enabled = (
        os.getenv(
            "ANALYTICS_WATCHDOG_ENABLED",
            "false",
        ).lower() == "true"
    )

    watchdog = None

    if watchdog_enabled:
        watchdog = WorkerWatchdog(
            check_interval=float(
                os.getenv(
                    "ANALYTICS_WATCHDOG_INTERVAL_SECONDS",
                    "5",
                )
            ),
            startup_timeout=float(
                os.getenv(
                    "ANALYTICS_WATCHDOG_STARTUP_TIMEOUT_SECONDS",
                    "120",
                )
            ),
            frame_timeout=float(
                os.getenv(
                    "ANALYTICS_WATCHDOG_FRAME_TIMEOUT_SECONDS",
                    "60",
                )
            ),
            reconnect_timeout=float(
                os.getenv(
                    "ANALYTICS_WATCHDOG_RECONNECT_TIMEOUT_SECONDS",
                    "0",
                )
            ),
        )

    app.state.analytics_worker = worker
    app.state.worker_watchdog = watchdog

    logger.info("Starting analytics worker")
    worker.start()

    try:
        if watchdog is not None:
            watchdog.start()

        yield

    finally:
        if watchdog is not None:
            logger.info("Stopping worker watchdog")

            if not watchdog.stop():
                logger.error(
                    "Watchdog failed to stop cleanly"
                )

        logger.info("Stopping analytics worker")

        stopped = worker.stop()

        if not stopped:
            logger.error(
                "Analytics worker failed to stop cleanly"
            )


app = FastAPI(
    title="Intelligent Video Analytics API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "intelligent-video-analytics",
    }

@app.get("/live")
def liveness():
    return {
        "status": "alive",
        "service": "intelligent-video-analytics",
    }

def get_database_health(request: Request) -> dict:
    worker = getattr(
        request.app.state,
        "analytics_worker",
        None,
    )

    if worker is not None:
        return worker.database_health()

    enabled = (
        os.getenv(
            "ANALYTICS_DATABASE_ENABLED",
            "false",
        ).lower() == "true"
    )

    return evaluate_writer_health(
        snapshot=None,
        enabled=enabled,
    )

@app.get("/ready")
def readiness(request: Request):
    database_health = get_database_health(request)

    result = evaluate_readiness(
        database_health=database_health,
    )

    return JSONResponse(
        status_code=(
            200 if result["status"] == "ready"
            else 503
        ),
        content=result,
    )

@app.get("/database/health")
def database_writer_health(request: Request):
    result = get_database_health(request)

    return JSONResponse(
        status_code=(
            503
            if result["healthy"] is False
            else 200
        ),
        content=result,
    )

@app.get("/status")
def status():
    return worker_state.snapshot()


@app.get("/metrics")
def metrics():
    snapshot = worker_state.snapshot()

    return {
        "frames_processed": snapshot["frames_processed"],
        "intrusion_events": snapshot["intrusion_events"],
        "throughput_fps": snapshot["throughput_fps"],
    }
    

def get_event_repository():
    """
    FastAPI dependency for database access.

    Keeping this as a dependency allows tests to
    replace PostgreSQL with a mock repository.
    """
    return create_event_repository()


@app.get(
    "/events",
    response_model=EventListResponse,
)
def list_events(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    model: str | None = None,
    zone: str | None = None,
    repository=Depends(get_event_repository),
):
    try:
        events, total = repository.list_events(
            limit=limit,
            offset=offset,
            model=model,
            zone=zone,
        )

        return EventListResponse(
            total=total,
            limit=limit,
            offset=offset,
            items=events,
        )

    except SQLAlchemyError:
        logger.exception(
            "Failed to retrieve intrusion events"
        )

        raise HTTPException(
            status_code=503,
            detail="Event database unavailable",
        )

@app.get("/prometheus", include_in_schema=False)
def prometheus_metrics(request: Request):
    health = get_database_health(request)

    snapshot = health["stats"]

    if snapshot is None:
        snapshot = {
            "saved": 0,
            "failed": 0,
            "dropped": 0,
            "pending": 0,
        }

    update_database_metrics(snapshot)
    update_database_health_metrics(health)

    return Response(
        content=render_metrics(),
        media_type=CONTENT_TYPE_LATEST,
    )