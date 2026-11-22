
from fastapi.testclient import TestClient

from video_analytics.api.app import app
from video_analytics.api.state import (
    WorkerState,
    WorkerStatus,
    worker_state,
)


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_initial_worker_state():
    state = WorkerState()
    snapshot = state.snapshot()

    assert snapshot["status"] == WorkerStatus.STOPPED
    assert snapshot["frames_processed"] == 0


def test_worker_state_updates():
    state = WorkerState()

    state.update(
        status=WorkerStatus.RUNNING,
        model="yolo",
        frames_processed=100,
        intrusion_events=2,
        throughput_fps=35.5,
    )

    snapshot = state.snapshot()

    assert snapshot["status"] == WorkerStatus.RUNNING
    assert snapshot["frames_processed"] == 100
    assert snapshot["intrusion_events"] == 2


def test_status_endpoint():
    worker_state.update(
        status=WorkerStatus.RUNNING,
        model="yolo",
        frames_processed=200,
    )

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json()["status"] == "RUNNING"
    assert response.json()["frames_processed"] == 200


def test_metrics_endpoint():
    worker_state.update(
        frames_processed=300,
        intrusion_events=3,
        throughput_fps=42.5,
    )

    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["frames_processed"] == 300
    assert data["intrusion_events"] == 3
    assert data["throughput_fps"] == 42.5
