
import numpy as np
import supervision as sv

from .detectors.base import DetectionResult


class ByteTrackTracker:
    def __init__(
        self,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(
        self,
        detections: DetectionResult,
    ) -> sv.Detections:
        sv_detections = sv.Detections(
            xyxy=detections.boxes.astype(np.float32),
            confidence=detections.scores.astype(np.float32),
            class_id=detections.class_ids.astype(int),
        )

        return self.tracker.update_with_detections(
            sv_detections
        )

    def reset(self):
        self.tracker.reset()
