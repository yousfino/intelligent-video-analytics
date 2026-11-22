
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class IntrusionEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    timestamp: datetime
    event_type: str
    zone: str
    track_id: int
    frame_number: int
    model: str
    created_at: datetime


class EventListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[IntrusionEventResponse]
