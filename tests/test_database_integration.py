
import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from video_analytics.database.connection import (
    create_db_engine,
)
from video_analytics.database.models import IntrusionEvent
from video_analytics.database.repository import (
    IntrusionEventRepository,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def database():
    # Prevent accidental writes to the development database.
    if os.getenv("POSTGRES_DB") != "video_analytics_test":
        pytest.skip(
            "Set POSTGRES_DB=video_analytics_test "
            "to run integration tests"
        )

    engine = create_db_engine()

    factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    try:
        yield factory
    finally:
        engine.dispose()


def sample_event():
    return {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "event_type": "zone_intrusion",
        "zone": f"integration_test_{uuid4().hex}",
        "track_id": 42,
        "frame_number": 100,
        "model": "yolo",
    }


def delete_test_record(factory, record_id):
    with factory.begin() as session:
        record = session.get(
            IntrusionEvent,
            record_id,
        )

        if record is not None:
            session.delete(record)


def test_real_database_insert(database):
    repository = IntrusionEventRepository(database)
    event = sample_event()

    record = repository.save(event)

    try:
        with database() as session:
            stored = session.get(
                IntrusionEvent,
                record.id,
            )

            assert stored is not None
            assert stored.zone == event["zone"]
            assert stored.track_id == 42
            assert stored.model == "yolo"
    finally:
        delete_test_record(database, record.id)


def test_real_database_query_and_filter(database):
    repository = IntrusionEventRepository(database)
    event = sample_event()

    record = repository.save(event)

    try:
        events, total = repository.list_events(
            limit=10,
            offset=0,
            model="yolo",
            zone=event["zone"],
        )

        assert total == 1
        assert len(events) == 1
        assert events[0].id == record.id

        empty, empty_total = repository.list_events(
            model="rfdetr",
            zone=event["zone"],
        )

        assert empty_total == 0
        assert empty == []
    finally:
        delete_test_record(database, record.id)


def test_failed_transaction_rolls_back(database):
    event_id = uuid4()

    with database() as session:
        invalid_record = IntrusionEvent(
            id=event_id,
            timestamp=datetime.now(timezone.utc),
            event_type="zone_intrusion",
            zone=None,  # Violates NOT NULL constraint
            track_id=1,
            frame_number=100,
            model="yolo",
        )

        session.add(invalid_record)

        with pytest.raises(IntegrityError):
            session.commit()

        session.rollback()

    with database() as session:
        result = session.get(
            IntrusionEvent,
            event_id,
        )

        assert result is None


def test_database_count_unchanged_after_cleanup(database):
    repository = IntrusionEventRepository(database)
    event = sample_event()

    with database() as session:
        before = session.scalar(
            select(func.count())
            .select_from(IntrusionEvent)
        )

    record = repository.save(event)

    delete_test_record(database, record.id)

    with database() as session:
        after = session.scalar(
            select(func.count())
            .select_from(IntrusionEvent)
        )

    assert before == after
