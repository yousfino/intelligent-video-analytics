
from fastapi import FastAPI

from video_analytics.api.state import worker_state


app = FastAPI(
    title="Intelligent Video Analytics API",
    version="0.1.0",
    description="Monitoring API for the video analytics worker",
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
