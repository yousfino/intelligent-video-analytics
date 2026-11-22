
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from video_analytics.api.app import (
    app,
    get_event_repository,
)


def sample_record():
    record = MagicMock()

    record.id = uuid4()
    record.timestamp = datetime.now(timezone.utc)
    record.event_type = "zone_intrusion"
    record.zone = "restricted_area_1"
    record.track_id = 7
    record.frame_number = 60
    record.model = "rfdetr"
    record.created_at = datetime.now(timezone.utc)

    return record


def test_list_events():
    repository = MagicMock()
    repository.list_events.return_value = (
        [sample_record()],
        1,
    )

    app.dependency_overrides[get_event_repository] = (
        lambda: repository
    )

    try:
        client = TestClient(app)
        response = client.get("/events")

        assert response.status_code == 200

        data = response.json()

        assert data["total"] == 1
        assert data["limit"] == 20
        assert data["offset"] == 0
        assert len(data["items"]) == 1
        assert data["items"][0]["model"] == "rfdetr"

    finally:
        app.dependency_overrides.clear()


def test_event_filters():
    repository = MagicMock()
    repository.list_events.return_value = ([], 0)

    app.dependency_overrides[get_event_repository] = (
        lambda: repository
    )

    try:
        client = TestClient(app)

        response = client.get(
            "/events?limit=5&offset=10"
            "&model=yolo&zone=restricted_area_1"
        )

        assert response.status_code == 200

        repository.list_events.assert_called_once_with(
            limit=5,
            offset=10,
            model="yolo",
            zone="restricted_area_1",
        )

    finally:
        app.dependency_overrides.clear()


def test_invalid_pagination():
    client = TestClient(app)

    assert client.get(
        "/events?limit=0"
    ).status_code == 422

    assert client.get(
        "/events?limit=101"
    ).status_code == 422

    assert client.get(
        "/events?offset=-1"
    ).status_code == 422


def test_database_unavailable():
    repository = MagicMock()

    repository.list_events.side_effect = OperationalError(
        "SELECT ...",
        {},
        Exception("Connection refused"),
    )

    app.dependency_overrides[get_event_repository] = (
        lambda: repository
    )

    try:
        client = TestClient(app)
        response = client.get("/events")

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "Event database unavailable"
        )

    finally:
        app.dependency_overrides.clear()
