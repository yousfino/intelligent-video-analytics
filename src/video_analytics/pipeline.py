
from dataclasses import dataclass

import numpy as np

from .detectors.base import BaseDetector
from .events import EventLogger
from .tracker import ByteTrackTracker
from .zones import RestrictedZone
import time

from .monitoring import PerformanceMonitor, PerformanceSnapshot


@dataclass
class PipelineResult:
    tracks: object
    inside_ids: set[int]
    events: list[dict]
    performance: PerformanceSnapshot | None = None


class AnalyticsPipeline:
    def __init__(
        self,
        detector: BaseDetector,
        tracker: ByteTrackTracker,
        zone: RestrictedZone,
        logger: EventLogger,
        model_name: str,
    ):
        self.detector = detector
        self.tracker = tracker
        self.zone = zone
        self.logger = logger
        self.model_name = model_name
        self.monitor = PerformanceMonitor(window_size=30)

    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> PipelineResult:
        height, width = frame.shape[:2]

        # Detection
        start = time.perf_counter()
        detections = self.detector.detect(frame)
        detection_ms = (
            time.perf_counter() - start
        ) * 1000.0

        # Tracking
        start = time.perf_counter()
        tracks = self.tracker.update(detections)
        tracking_ms = (
            time.perf_counter() - start
        ) * 1000.0

        # Zone analytics and event logging
        start = time.perf_counter()

        inside_ids = set()

        if tracks.tracker_id is not None:
            for box, track_id in zip(
                tracks.xyxy, tracks.tracker_id
            ):
                x1, y1, x2, y2 = map(int, box)

                foot_point = (
                    int((x1 + x2) / 2),
                    y2,
                )

                if self.zone.contains(
                    foot_point, width, height
                ):
                    inside_ids.add(int(track_id))

        entered_ids = self.zone.update(inside_ids)

        events = []

        for track_id in sorted(entered_ids):
            event = self.logger.log_intrusion(
                zone=self.zone.name,
                track_id=track_id,
                frame_number=frame_number,
                model=self.model_name,
            )
            events.append(event)

        analytics_ms = (
            time.perf_counter() - start
        ) * 1000.0

        performance = self.monitor.record(
            detection_ms=detection_ms,
            tracking_ms=tracking_ms,
            analytics_ms=analytics_ms,
        )

        return PipelineResult(
            tracks=tracks,
            inside_ids=inside_ids,
            events=events,
            performance=performance,
        )


    def reset(self):
        self.tracker.reset()
        self.zone.active_ids.clear()
