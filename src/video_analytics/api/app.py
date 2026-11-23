
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

    app.state.analytics_worker = worker

    logger.info("Starting analytics worker")
    worker.start()

    try:
        yield
    finally:
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
def prometheus_metrics():
    return Response(
        content=render_metrics(),
        media_type=CONTENT_TYPE_LATEST,
    )