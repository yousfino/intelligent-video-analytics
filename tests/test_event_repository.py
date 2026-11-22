
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from video_analytics.database.repository import (
    IntrusionEventRepository,
)


def sample_event():
    return {
        "timestamp": "2026-11-22T11:12:49+00:00",
        "event_type": "zone_intrusion",
        "zone": "restricted_area_1",
        "track_id": 7,
        "frame_number": 60,
        "model": "rfdetr",
    }


def test_repository_saves_event():
    session = MagicMock()

    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = (
        session
    )

    repository = IntrusionEventRepository(
        session_factory
    )

    record = repository.save(sample_event())

    assert record.event_type == "zone_intrusion"
    assert record.zone == "restricted_area_1"
    assert record.track_id == 7
    assert record.frame_number == 60
    assert record.model == "rfdetr"

    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(record)


def test_repository_accepts_utc_timestamp():
    session = MagicMock()

    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = (
        session
    )

    repository = IntrusionEventRepository(
        session_factory
    )

    record = repository.save(sample_event())

    assert record.timestamp == datetime(
        2026, 11, 22, 11, 12, 49,
        tzinfo=timezone.utc,
    )


def test_repository_propagates_database_failure():
    session = MagicMock()
    session.commit.side_effect = RuntimeError(
        "Database unavailable"
    )

    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = (
        session
    )

    repository = IntrusionEventRepository(
        session_factory
    )

    with pytest.raises(
        RuntimeError,
        match="Database unavailable",
    ):
        repository.save(sample_event())
