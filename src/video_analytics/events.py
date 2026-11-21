
import json
from datetime import datetime, timezone
from pathlib import Path


class EventLogger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

    def log_intrusion(
        self,
        zone: str,
        track_id: int,
        frame_number: int,
        model: str,
    ) -> dict:
        event = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "event_type": "zone_intrusion",
            "zone": zone,
            "track_id": int(track_id),
            "frame_number": int(frame_number),
            "model": model,
        }

        with self.output_path.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(event) + "\n")

        return event
