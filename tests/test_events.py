
import json

from video_analytics.events import EventLogger


def test_intrusion_event_logging(tmp_path):
    output = tmp_path / "events.jsonl"

    logger = EventLogger(str(output))

    event = logger.log_intrusion(
        zone="test_zone",
        track_id=7,
        frame_number=42,
        model="yolo",
    )

    assert output.exists()

    saved = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert saved == event
    assert saved["event_type"] == "zone_intrusion"
    assert saved["track_id"] == 7
    assert saved["frame_number"] == 42
    assert saved["model"] == "yolo"
