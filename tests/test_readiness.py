
import time

import pytest
from fastapi.testclient import TestClient

from video_analytics.api.app import app
from video_analytics.api.state import (
    WorkerStatus,
    worker_state,
)


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_readiness_state(monkeypatch):
    original = worker_state.snapshot()

    monkeypatch.setenv(
        "ANALYTICS_DATABASE_ENABLED",
        "false",
    )

    monkeypatch.setenv(
        "ANALYTICS_MAX_FRAME_AGE_SECONDS",
        "15",
    )

    worker_state.update(
        status=WorkerStatus.RUNNING,
        frames_processed=100,
        last_frame_monotonic=time.monotonic(),
        error=None,
    )

    yield

    worker_state.update(**original)


def test_liveness():
    response = client.get("/live")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_with_active_worker():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.parametrize(
    "status",
    [
        WorkerStatus.STOPPED,
        WorkerStatus.STARTING,
        WorkerStatus.RECONNECTING,
        WorkerStatus.STOPPING,
        WorkerStatus.ERROR,
    ],
)
def test_not_ready_with_inactive_worker(status):
    worker_state.update(status=status)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert not response.json()["checks"]["worker_running"]


def test_not_ready_without_processed_frames():
    worker_state.update(
        frames_processed=0,
        last_frame_monotonic=None,
    )

    response = client.get("/ready")

    assert response.status_code == 503

    assert not response.json()["checks"][
        "frame_processing"
    ]


def test_not_ready_with_stale_heartbeat():
    worker_state.update(
        last_frame_monotonic=time.monotonic() - 60,
    )

    response = client.get("/ready")

    assert response.status_code == 503

    assert not response.json()["checks"][
        "frame_processing"
    ]


def test_ready_with_database_available(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_DATABASE_ENABLED",
        "true",
    )

    monkeypatch.setattr(
        "video_analytics.api.readiness."
        "check_database_connection",
        lambda: "video_analytics_test",
    )

    response = client.get("/ready")

    assert response.status_code == 200

    assert response.json()["checks"][
        "database_available"
    ] is True


def test_not_ready_with_database_unavailable(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_DATABASE_ENABLED",
        "true",
    )

    def unavailable():
        raise ConnectionError("Database unavailable")

    monkeypatch.setattr(
        "video_analytics.api.readiness."
        "check_database_connection",
        unavailable,
    )

    response = client.get("/ready")

    assert response.status_code == 503

    assert response.json()["checks"][
        "database_available"
    ] is False


def test_liveness_survives_worker_failure():
    worker_state.update(
        status=WorkerStatus.ERROR,
    )

    assert client.get("/live").status_code == 200
    assert client.get("/ready").status_code == 503
