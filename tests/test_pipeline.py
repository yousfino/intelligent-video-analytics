
import json

import numpy as np

from video_analytics.detectors.base import (
    BaseDetector,
    DetectionResult,
)
from video_analytics.events import EventLogger
from video_analytics.pipeline import AnalyticsPipeline
from video_analytics.tracker import ByteTrackTracker
from video_analytics.zones import RestrictedZone


class MockDetector(BaseDetector):
    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def detect(self, frame):
        return next(self.sequence)


def make_detection(box):
    return DetectionResult(
        boxes=[box],
        scores=[0.95],
        class_ids=[0],
    )


def test_intrusion_logged_once(tmp_path):
    # Restricted zone covers the right half.
    zone = RestrictedZone(
        name="test_zone",
        normalized_points=[
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 1.0],
        ],
    )

    # 200 x 200 simulated frames.
    frame = np.zeros(
        (200, 200, 3), dtype=np.uint8
    )

    # Person moves into the restricted zone.
    sequence = [
        make_detection([20, 40, 60, 150]),
        make_detection([30, 40, 70, 150]),
        make_detection([50, 40, 90, 150]),
        make_detection([80, 40, 120, 150]),
        make_detection([90, 40, 130, 150]),
        make_detection([95, 40, 135, 150]),
    ]

    detector = MockDetector(sequence)
    tracker = ByteTrackTracker()
    logger = EventLogger(
        str(tmp_path / "events.jsonl")
    )

    pipeline = AnalyticsPipeline(
        detector=detector,
        tracker=tracker,
        zone=zone,
        logger=logger,
        model_name="mock",
    )

    all_events = []

    for frame_number in range(1, 7):
        result = pipeline.process_frame(
            frame, frame_number
        )
        all_events.extend(result.events)

    assert len(all_events) == 1
    assert all_events[0]["event_type"] == (
        "zone_intrusion"
    )
    assert all_events[0]["zone"] == "test_zone"

    lines = (
        tmp_path / "events.jsonl"
    ).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0]) == all_events[0]


def test_empty_frame_does_not_log_event(tmp_path):
    zone = RestrictedZone(
        name="test_zone",
        normalized_points=[
            [0.5, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 1.0],
        ],
    )

    detector = MockDetector([
        DetectionResult(
            boxes=[],
            scores=[],
            class_ids=[],
        )
    ])

    pipeline = AnalyticsPipeline(
        detector=detector,
        tracker=ByteTrackTracker(),
        zone=zone,
        logger=EventLogger(
            str(tmp_path / "events.jsonl")
        ),
        model_name="mock",
    )

    frame = np.zeros(
        (200, 200, 3), dtype=np.uint8
    )

    result = pipeline.process_frame(
        frame, frame_number=1
    )

    assert len(result.tracks) == 0
    assert result.inside_ids == set()
    assert result.events == []
