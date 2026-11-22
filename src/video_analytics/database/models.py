
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class IntrusionEvent(Base):
    __tablename__ = "intrusion_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    zone: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    track_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    frame_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    model: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_intrusion_events_timestamp",
            "timestamp",
        ),
        Index(
            "ix_intrusion_events_zone",
            "zone",
        ),
    )
