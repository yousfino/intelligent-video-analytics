
import pytest

from video_analytics.api.state import (
    WorkerStatus,
    worker_state,
)

from video_analytics.api.worker import AnalyticsWorker


@pytest.fixture(autouse=True)
def restore_worker_state():
    original = worker_state.snapshot()

    worker_state.update(
        status=WorkerStatus.RUNNING,
        frames_processed=100,
        stream_connected=True,
        reconnect_attempts=0,
        total_reconnections=0,
    )

    yield

    worker_state.update(**original)


def test_worker_reports_reconnection():
    worker = AnalyticsWorker(source=0)

    worker._on_stream_state_change(
        "RECONNECTING",
        2,
    )

    snapshot = worker_state.snapshot()

    assert snapshot["status"] == WorkerStatus.RECONNECTING
    assert snapshot["stream_connected"] is False
    assert snapshot["reconnect_attempts"] == 2
    assert snapshot["last_frame_monotonic"] is None


def test_worker_reports_recovered_stream():
    worker = AnalyticsWorker(source=0)

    worker._on_stream_state_change(
        "RECONNECTING",
        1,
    )

    worker._on_stream_state_change(
        "CONNECTED",
        1,
    )

    snapshot = worker_state.snapshot()

    assert snapshot["stream_connected"] is True
    assert snapshot["reconnect_attempts"] == 0
    assert snapshot["total_reconnections"] == 1

    # Inference must succeed before RUNNING is restored.
    assert snapshot["status"] == WorkerStatus.RECONNECTING


def test_initial_connection_is_not_reconnection():
    worker = AnalyticsWorker(source=0)

    worker_state.update(
        status=WorkerStatus.STARTING,
        frames_processed=0,
        stream_connected=False,
    )

    worker._on_stream_state_change(
        "CONNECTED",
        0,
    )

    snapshot = worker_state.snapshot()

    assert snapshot["stream_connected"] is True
    assert snapshot["total_reconnections"] == 0


def test_worker_ignores_callback_during_shutdown():
    worker = AnalyticsWorker(source=0)

    worker.stop_event.set()

    worker_state.update(
        status=WorkerStatus.STOPPING,
    )

    worker._on_stream_state_change(
        "RECONNECTING",
        1,
    )

    assert (
        worker_state.snapshot()["status"]
        == WorkerStatus.STOPPING
    )
