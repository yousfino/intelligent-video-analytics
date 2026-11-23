
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from video_analytics.api.app import app
from video_analytics.api.state import (
    WorkerStatus,
    worker_state,
)
from video_analytics.api.worker import AnalyticsWorker


def test_worker_starts_on_api_startup(monkeypatch):
    monkeypatch.delenv("ANALYTICS_CONFIG", raising=False)
    monkeypatch.setenv("ANALYTICS_SOURCE", "0")
    monkeypatch.setenv("ANALYTICS_MODEL", "yolo")
    monkeypatch.setenv("ANALYTICS_LOOP", "false")

    with patch(
        "video_analytics.api.app.AnalyticsWorker"
    ) as mock_worker_class:
        mock_worker = mock_worker_class.return_value
        mock_worker.stop.return_value = True

        with TestClient(app) as client:
            mock_worker_class.assert_called_once_with(
                source=0,
                model="yolo",
                loop_file=False,
                config_path="configs/pipeline.yaml",
            )

            mock_worker.start.assert_called_once()

            response = client.get("/health")
            assert response.status_code == 200


def test_worker_stops_on_api_shutdown(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_SOURCE",
        "inputs/test_video.mp4",
    )
    monkeypatch.setenv("ANALYTICS_MODEL", "rfdetr")
    monkeypatch.setenv("ANALYTICS_LOOP", "true")

    with patch(
        "video_analytics.api.app.AnalyticsWorker"
    ) as mock_worker_class:
        mock_worker = mock_worker_class.return_value
        mock_worker.stop.return_value = True

        with TestClient(app):
            mock_worker.start.assert_called_once()
            mock_worker.stop.assert_not_called()

        mock_worker.stop.assert_called_once()


def test_worker_records_initialization_failure():
    worker_state.update(
        status=WorkerStatus.STOPPED,
        error=None,
    )

    worker = AnalyticsWorker(
        source="invalid.mp4",
        model="yolo",
    )

    with patch(
        "video_analytics.api.worker.load_config",
        side_effect=RuntimeError("Test initialization failure"),
    ):
        worker._run()

    snapshot = worker_state.snapshot()

    assert snapshot["status"] == WorkerStatus.ERROR
    assert snapshot["error"] == (
        "Test initialization failure"
    )

    # Restore shared state for other tests.
    worker_state.update(
        status=WorkerStatus.STOPPED,
        error=None,
    )
