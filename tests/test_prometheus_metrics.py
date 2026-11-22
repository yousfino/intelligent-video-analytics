from fastapi.testclient import TestClient

from video_analytics.api.app import app
from video_analytics.api.prometheus_metrics import (
    update_worker_metrics,
    update_database_metrics,
)


def test_prometheus_endpoint():
    update_worker_metrics(
        frames=100,
        intrusions=3,
        fps=25.5,
    )

    update_database_metrics({
        "saved": 3,
        "failed": 0,
        "dropped": 0,
        "pending": 1,
    })

    client = TestClient(app)
    response = client.get("/prometheus")

    assert response.status_code == 200

    body = response.text

    assert "iva_frames_processed 100.0" in body
    assert "iva_intrusion_events 3.0" in body
    assert "iva_throughput_fps 25.5" in body
    assert "iva_db_events_saved 3.0" in body
    assert "iva_db_queue_depth 1.0" in body
