
import threading

from unittest.mock import MagicMock

from video_analytics.database.writer import AsyncEventWriter


def sample_event():
    return {
        "timestamp": "2026-11-22T13:40:21+00:00",
        "event_type": "zone_intrusion",
        "zone": "restricted_area_1",
        "track_id": 1,
        "frame_number": 60,
        "model": "rfdetr",
    }


def test_async_writer_saves_events():
    repository = MagicMock()
    writer = AsyncEventWriter(repository, retry_delay=0)

    writer.start()
    assert writer.submit(sample_event())
    assert writer.stop()

    repository.save.assert_called_once()
    assert writer.snapshot()["saved"] == 1
    assert writer.snapshot()["pending"] == 0


def test_async_writer_retries_failure():
    repository = MagicMock()
    repository.save.side_effect = [
        RuntimeError("Temporary failure"),
        None,
    ]

    writer = AsyncEventWriter(
        repository,
        max_attempts=3,
        retry_delay=0,
    )

    writer.start()
    writer.submit(sample_event())
    assert writer.stop()

    assert repository.save.call_count == 2
    assert writer.snapshot()["saved"] == 1
    assert writer.snapshot()["failed"] == 0


def test_async_writer_records_permanent_failure():
    repository = MagicMock()
    repository.save.side_effect = RuntimeError(
        "Database unavailable"
    )

    writer = AsyncEventWriter(
        repository,
        max_attempts=3,
        retry_delay=0,
    )

    writer.start()
    writer.submit(sample_event())
    assert writer.stop()

    assert repository.save.call_count == 3
    assert writer.snapshot()["failed"] == 1


def test_async_writer_queue_capacity():
    repository = MagicMock()
    writer = AsyncEventWriter(
        repository,
        max_queue_size=1,
    )

    # Writer has not started, so submission is rejected.
    assert writer.submit(sample_event()) is False

    # Start with the worker blocked on a database write.
    entered = threading.Event()
    release = threading.Event()

    def blocked_save(event):
        entered.set()
        assert release.wait(timeout=5)

    repository.save.side_effect = blocked_save

    writer.start()

    try:
        assert writer.submit(sample_event())
        assert entered.wait(timeout=5)

        assert writer.submit(sample_event())
        assert writer.submit(sample_event()) is False

        assert writer.snapshot()["dropped"] == 1
    finally:
        release.set()
        assert writer.stop()


def test_async_writer_cleanup():
    repository = MagicMock()
    cleanup = MagicMock()

    writer = AsyncEventWriter(
        repository,
        retry_delay=0,
        on_close=cleanup,
    )

    writer.start()
    assert writer.stop()

    cleanup.assert_called_once()
