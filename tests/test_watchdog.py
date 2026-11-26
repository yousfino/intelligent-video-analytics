
from unittest.mock import MagicMock

import pytest

from video_analytics.api.state import WorkerStatus
from video_analytics.api.watchdog import WorkerWatchdog


@pytest.fixture
def watchdog_context():
    now = [100.0]

    snapshot = {
        "status": WorkerStatus.RUNNING,
        "last_frame_monotonic": 99.0,
        "error": None,
    }

    failure_handler = MagicMock()

    watchdog = WorkerWatchdog(
        check_interval=1.0,
        startup_timeout=120.0,
        frame_timeout=60.0,
        snapshot_provider=lambda: snapshot.copy(),
        on_failure=failure_handler,
        clock=lambda: now[0],
    )

    return watchdog, snapshot, now, failure_handler


def test_healthy_worker(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    watchdog.check_once()

    handler.assert_not_called()


def test_worker_error(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["status"] = WorkerStatus.ERROR
    snapshot["error"] = "Model initialization failed"

    watchdog.check_once()

    handler.assert_called_once_with(
        "Analytics worker entered ERROR: "
        "Model initialization failed"
    )


def test_unexpected_worker_stop(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["status"] = WorkerStatus.STOPPED

    watchdog.check_once()

    handler.assert_called_once_with(
        "Analytics worker stopped unexpectedly"
    )


def test_startup_within_timeout(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["status"] = WorkerStatus.STARTING
    now[0] = 150.0

    watchdog.check_once()

    handler.assert_not_called()


def test_startup_timeout(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["status"] = WorkerStatus.STARTING
    now[0] = 221.0

    watchdog.check_once()

    handler.assert_called_once_with(
        "Analytics worker startup timed out"
    )


def test_stale_frame_heartbeat(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    now[0] = 161.0

    watchdog.check_once()

    handler.assert_called_once_with(
        "Frame processing heartbeat timed out"
    )


def test_missing_frame_heartbeat(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["last_frame_monotonic"] = None

    watchdog.check_once()

    handler.assert_called_once_with(
        "Running worker has no frame heartbeat"
    )


def test_unlimited_reconnection(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["status"] = WorkerStatus.RECONNECTING

    watchdog.check_once()

    now[0] += 10000

    watchdog.check_once()

    handler.assert_not_called()


def test_finite_reconnection_timeout(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    watchdog.reconnect_timeout = 30.0

    snapshot["status"] = WorkerStatus.RECONNECTING

    watchdog.check_once()

    now[0] += 31.0

    watchdog.check_once()

    handler.assert_called_once_with(
        "Video reconnection timed out"
    )


def test_reconnection_timer_resets(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    watchdog.reconnect_timeout = 30.0

    snapshot["status"] = WorkerStatus.RECONNECTING
    watchdog.check_once()

    now[0] += 20.0

    snapshot["status"] = WorkerStatus.RUNNING
    snapshot["last_frame_monotonic"] = now[0]

    watchdog.check_once()

    now[0] += 20.0

    snapshot["status"] = WorkerStatus.RECONNECTING

    watchdog.check_once()

    handler.assert_not_called()


def test_intentional_shutdown(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["status"] = WorkerStatus.STOPPING

    watchdog.check_once()

    handler.assert_not_called()


def test_failure_reported_once(watchdog_context):
    watchdog, snapshot, now, handler = watchdog_context

    snapshot["status"] = WorkerStatus.ERROR

    watchdog.check_once()
    watchdog.check_once()

    assert handler.call_count == 1


def test_stopped_watchdog_does_not_trigger(
    watchdog_context,
):
    watchdog, snapshot, now, handler = watchdog_context

    watchdog.stop()

    snapshot["status"] = WorkerStatus.ERROR

    watchdog.check_once()

    handler.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"check_interval": 0},
        {"startup_timeout": -1},
        {"frame_timeout": float("nan")},
        {"reconnect_timeout": -1},
    ],
)
def test_invalid_watchdog_configuration(settings):
    with pytest.raises(ValueError):
        WorkerWatchdog(**settings)


def test_watchdog_thread_lifecycle():
    watchdog = WorkerWatchdog(
        check_interval=60.0,
        on_failure=MagicMock(),
    )

    watchdog.start()

    assert watchdog.thread.is_alive()

    assert watchdog.stop(timeout=2.0)

    assert not watchdog.thread.is_alive()
