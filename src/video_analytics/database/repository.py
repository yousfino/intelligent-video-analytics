
import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from video_analytics.database.connection import create_db_engine
from video_analytics.database.models import IntrusionEvent
from sqlalchemy import func, select

logger = logging.getLogger(__name__)


class IntrusionEventRepository:
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def save(self, event: dict[str, Any]) -> IntrusionEvent:
        timestamp = datetime.fromisoformat(
            event["timestamp"].replace("Z", "+00:00")
        )

        record = IntrusionEvent(
            timestamp=timestamp,
            event_type=event["event_type"],
            zone=event["zone"],
            track_id=int(event["track_id"]),
            frame_number=int(event["frame_number"]),
            model=event["model"],
        )

        with self.session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)

        return record
    
    
    def list_events(
        self,
        limit: int = 20,
        offset: int = 0,
        model: str | None = None,
        zone: str | None = None,
    ) -> tuple[list[IntrusionEvent], int]:
        filters = []

        if model is not None:
            filters.append(IntrusionEvent.model == model)

        if zone is not None:
            filters.append(IntrusionEvent.zone == zone)

        count_query = (
            select(func.count())
            .select_from(IntrusionEvent)
            .where(*filters)
        )

        events_query = (
            select(IntrusionEvent)
            .where(*filters)
            .order_by(
                IntrusionEvent.timestamp.desc(),
                IntrusionEvent.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        with self.session_factory() as session:
            total = session.scalar(count_query) or 0
            events = list(
                session.scalars(events_query).all()
            )

        return events, total



def create_event_repository():
    engine = create_db_engine()

    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    return IntrusionEventRepository(session_factory)
