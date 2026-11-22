
from datetime import datetime, timezone

from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql

from video_analytics.database.models import (
    Base,
    IntrusionEvent,
)


def test_intrusion_events_table_exists():
    assert "intrusion_events" in Base.metadata.tables


def test_intrusion_event_columns():
    table = IntrusionEvent.__table__

    expected_columns = {
        "id",
        "timestamp",
        "event_type",
        "zone",
        "track_id",
        "frame_number",
        "model",
        "created_at",
    }

    assert set(table.columns.keys()) == expected_columns


def test_intrusion_event_primary_key():
    table = IntrusionEvent.__table__

    primary_keys = {
        column.name
        for column in table.primary_key.columns
    }

    assert primary_keys == {"id"}


def test_intrusion_event_indexes():
    table = IntrusionEvent.__table__

    indexes = {
        index.name
        for index in table.indexes
    }

    assert "ix_intrusion_events_timestamp" in indexes
    assert "ix_intrusion_events_zone" in indexes


def test_intrusion_event_postgresql_ddl():
    ddl = str(
        CreateTable(
            IntrusionEvent.__table__
        ).compile(
            dialect=postgresql.dialect()
        )
    )

    assert "CREATE TABLE intrusion_events" in ddl
    assert "UUID" in ddl
    assert "TIMESTAMP WITH TIME ZONE" in ddl


def test_intrusion_event_instance():
    event = IntrusionEvent(
        timestamp=datetime.now(timezone.utc),
        event_type="zone_intrusion",
        zone="restricted_area_1",
        track_id=1,
        frame_number=60,
        model="rfdetr",
    )

    assert event.event_type == "zone_intrusion"
    assert event.track_id == 1
    assert event.model == "rfdetr"
