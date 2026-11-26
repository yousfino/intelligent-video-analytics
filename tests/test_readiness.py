
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
        stream_connected=True,
        reconnect_attempts=0,
        total_reconnections=0,
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

    monkeypatch.setattr(
        "video_analytics.api.app.get_database_health",
        lambda request: {
            "enabled": True,
            "healthy": True,
            "reasons": [],
            "stats": {},
        },
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


def test_not_ready_with_disconnected_stream():
    worker_state.update(
        stream_connected=False,
    )

    response = client.get("/ready")

    assert response.status_code == 503

    assert response.json()["checks"][
        "stream_connected"
    ] is False


def test_readiness_recovers_after_reconnection():
    worker_state.update(
        status=WorkerStatus.RECONNECTING,
        stream_connected=False,
        last_frame_monotonic=None,
        reconnect_attempts=2,
    )

    assert client.get("/ready").status_code == 503
    assert client.get("/live").status_code == 200

    worker_state.update(
        status=WorkerStatus.RUNNING,
        stream_connected=True,
        last_frame_monotonic=time.monotonic(),
        reconnect_attempts=0,
        total_reconnections=1,
    )

    assert client.get("/ready").status_code == 200


def test_stream_reconnection_state_in_status():
    worker_state.update(
        status=WorkerStatus.RECONNECTING,
        stream_connected=False,
        reconnect_attempts=3,
        total_reconnections=2,
    )

    response = client.get("/status")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "RECONNECTING"
    assert data["stream_connected"] is False
    assert data["reconnect_attempts"] == 3
    assert data["total_reconnections"] == 2


def test_not_ready_with_unhealthy_writer(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_DATABASE_ENABLED",
        "true",
    )

    monkeypatch.setattr(
        "video_analytics.api.readiness."
        "check_database_connection",
        lambda: "video_analytics_test",
    )

    monkeypatch.setattr(
        "video_analytics.api.app.get_database_health",
        lambda request: {
            "enabled": True,
            "healthy": False,
            "reasons": [
                "Database writer heartbeat is stale"
            ],
            "stats": {},
        },
    )

    response = client.get("/ready")

    assert response.status_code == 503

    checks = response.json()["checks"]

    assert checks["database_available"] is True
    assert checks["database_writer_healthy"] is False


def test_database_health_endpoint_disabled():
    response = client.get("/database/health")

    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_database_health_endpoint_unavailable(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_DATABASE_ENABLED",
        "true",
    )

    monkeypatch.setattr(
        "video_analytics.api.app.get_database_health",
        lambda request: {
            "enabled": True,
            "healthy": False,
            "reasons": ["Database writer unavailable"],
            "stats": None,
        },
    )

    response = client.get("/database/health")

    assert response.status_code == 503
    assert response.json()["healthy"] is False
