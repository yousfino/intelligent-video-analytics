
import numpy as np

from video_analytics.detectors.base import DetectionResult
from video_analytics.tracker import ByteTrackTracker


def test_tracker_handles_empty_detections():
    tracker = ByteTrackTracker()

    detections = DetectionResult(
        boxes=[],
        scores=[],
        class_ids=[],
    )

    tracks = tracker.update(detections)

    assert len(tracks) == 0


def test_tracker_assigns_ids():
    tracker = ByteTrackTracker()

    detections = DetectionResult(
        boxes=[[100, 100, 200, 300]],
        scores=[0.95],
        class_ids=[0],
    )

    # ByteTrack may need multiple frames to activate
    # and confirm a track.
    tracks = None

    for _ in range(5):
        tracks = tracker.update(detections)

    assert len(tracks) == 1
    assert tracks.tracker_id is not None
    assert tracks.tracker_id[0] >= 0


def test_tracker_reset():
    tracker = ByteTrackTracker()

    detections = DetectionResult(
        boxes=[[100, 100, 200, 300]],
        scores=[0.95],
        class_ids=[0],
    )

    tracker.update(detections)
    tracker.reset()

    empty = DetectionResult(
        boxes=[],
        scores=[],
        class_ids=[],
    )

    tracks = tracker.update(empty)

    assert len(tracks) == 0
