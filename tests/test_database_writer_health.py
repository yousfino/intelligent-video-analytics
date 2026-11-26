
import threading
import time

from unittest.mock import MagicMock

import pytest

from video_analytics.database.writer import AsyncEventWriter
from video_analytics.database.health import evaluate_writer_health


@pytest.fixture
def healthy_snapshot():
    return {
        "alive": True,
        "stopping": False,
        "fatal_error": None,
        "heartbeat_age_seconds": 0.1,
        "queue_utilization": 0.0,
        "consecutive_failures": 0,
    }


def test_disabled_database_writer():
    result = evaluate_writer_health(None, enabled=False)

    assert result["enabled"] is False
    assert result["healthy"] is None


def test_missing_database_writer():
    result = evaluate_writer_health(None, enabled=True)

    assert result["healthy"] is False


def test_healthy_database_writer(healthy_snapshot):
    result = evaluate_writer_health(
        healthy_snapshot,
        enabled=True,
    )

    assert result["healthy"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("alive", False),
        ("stopping", True),
        ("fatal_error", "Unexpected failure"),
        ("heartbeat_age_seconds", None),
        ("heartbeat_age_seconds", 31),
        ("queue_utilization", 1.0),
        ("consecutive_failures", 3),
    ],
)
def test_unhealthy_database_writer(
    healthy_snapshot,
    field,
    value,
):
    healthy_snapshot[field] = value

    result = evaluate_writer_health(
        healthy_snapshot,
        enabled=True,
        max_heartbeat_age=30,
        max_consecutive_failures=3,
    )

    assert result["healthy"] is False
    assert result["reasons"]


def test_idle_writer_remains_healthy():
    writer = AsyncEventWriter(MagicMock())

    writer.start()

    try:
        time.sleep(0.3)

        snapshot = writer.snapshot()

        assert snapshot["alive"] is True
        assert snapshot["saved"] == 0

        result = evaluate_writer_health(
            snapshot,
            enabled=True,
            max_heartbeat_age=2,
        )

        assert result["healthy"] is True

    finally:
        assert writer.stop()


def test_blocked_writer_detected():
    repository = MagicMock()

    entered = threading.Event()
    release = threading.Event()

    def blocked_save(event):
        entered.set()
        release.wait(timeout=5)

    repository.save.side_effect = blocked_save

    writer = AsyncEventWriter(repository)

    writer.start()

    try:
        assert writer.submit({"event": "test"})
        assert entered.wait(timeout=2)

        time.sleep(0.2)

        snapshot = writer.snapshot()

        assert snapshot["in_flight"] is True

        result = evaluate_writer_health(
            snapshot,
            enabled=True,
            max_heartbeat_age=0.1,
        )

        assert result["healthy"] is False

        assert (
            "Database writer heartbeat is stale"
            in result["reasons"]
        )

    finally:
        release.set()
        assert writer.stop()


def test_writer_recovers_after_failed_event():
    repository = MagicMock()

    repository.save.side_effect = [
        RuntimeError("Database unavailable"),
        None,
    ]

    writer = AsyncEventWriter(
        repository,
        max_attempts=1,
        retry_delay=0,
    )

    writer.start()

    try:
        assert writer.submit({"event": 1})

        writer.events.join()

        assert writer.snapshot()["consecutive_failures"] == 1

        assert writer.submit({"event": 2})

        writer.events.join()

        snapshot = writer.snapshot()

        assert snapshot["saved"] == 1
        assert snapshot["failed"] == 1
        assert snapshot["consecutive_failures"] == 0
        assert snapshot["last_error"] is None

    finally:
        assert writer.stop()


def test_writer_rejects_submission_after_shutdown():
    writer = AsyncEventWriter(MagicMock())

    writer.start()

    assert writer.stop()

    assert writer.submit({"event": "test"}) is False

    assert writer.snapshot()["dropped"] == 1


def test_writer_fatal_failure():
    repository = MagicMock()

    repository.save.side_effect = SystemExit(
        "Simulated thread failure"
    )

    writer = AsyncEventWriter(repository)

    writer.start()

    try:
        assert writer.submit({"event": "test"})

        writer.thread.join(timeout=2)

        snapshot = writer.snapshot()

        assert snapshot["alive"] is False
        assert snapshot["fatal_error"] is not None

        assert writer.submit({"event": "test"}) is False

        result = evaluate_writer_health(
            snapshot,
            enabled=True,
        )

        assert result["healthy"] is False

    finally:
        assert writer.stop()
