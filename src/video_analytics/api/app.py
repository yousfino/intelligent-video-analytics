
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from video_analytics.api.state import worker_state
from video_analytics.api.worker import AnalyticsWorker

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
