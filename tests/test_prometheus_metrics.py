
from fastapi.testclient import TestClient

from video_analytics.api.app import app
from video_analytics.api.prometheus_metrics import (
    update_worker_metrics,
)


def test_prometheus_endpoint(monkeypatch):
    update_worker_metrics(
        frames=100,
        intrusions=3,
        fps=25.5,
    )

    monkeypatch.setattr(
        "video_analytics.api.app.get_database_health",
        lambda request: {
            "enabled": True,
            "healthy": True,
            "reasons": [],
            "stats": {
                "saved": 3,
                "failed": 0,
                "dropped": 0,
                "pending": 1,
                "alive": True,
                "stopping": False,
                "in_flight": False,
                "queue_capacity": 1000,
                "queue_utilization": 0.001,
                "heartbeat_age_seconds": 0.05,
                "last_success_age_seconds": 1.0,
                "consecutive_failures": 0,
                "last_error": None,
                "fatal_error": None,
            },
        },
    )

    client = TestClient(app)
    response = client.get("/prometheus")

    assert response.status_code == 200

    body = response.text

    # Existing analytics metrics
    assert "iva_frames_processed 100.0" in body
    assert "iva_intrusion_events 3.0" in body
    assert "iva_throughput_fps 25.5" in body

    # Existing database metrics
    assert "iva_db_events_saved 3.0" in body
    assert "iva_db_events_failed 0.0" in body
    assert "iva_db_events_dropped 0.0" in body
    assert "iva_db_queue_depth 1.0" in body

    # New database writer health metrics
    assert "iva_db_writer_enabled 1.0" in body
    assert "iva_db_writer_alive 1.0" in body
    assert "iva_db_writer_healthy 1.0" in body
    assert "iva_db_writer_heartbeat_age_seconds 0.05" in body
    assert "iva_db_queue_utilization 0.001" in body
    assert "iva_db_consecutive_failures 0.0" in body
    assert "iva_db_write_in_flight 0.0" in body


def test_prometheus_database_disabled(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_DATABASE_ENABLED",
        "false",
    )

    client = TestClient(app)
    response = client.get("/prometheus")

    assert response.status_code == 200

    body = response.text

    assert "iva_db_writer_enabled 0.0" in body
    assert "iva_db_writer_alive 0.0" in body
    assert "iva_db_writer_healthy 0.0" in body
    assert "iva_db_events_saved 0.0" in body
    assert "iva_db_events_failed 0.0" in body
    assert "iva_db_events_dropped 0.0" in body
    assert "iva_db_queue_depth 0.0" in body
    assert "iva_db_writer_heartbeat_age_seconds NaN" in body
