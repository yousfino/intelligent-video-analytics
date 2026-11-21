
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSnapshot:
    detection_ms: float
    tracking_ms: float
    analytics_ms: float
    pipeline_ms: float
    pipeline_fps: float


class PerformanceMonitor:
    def __init__(self, window_size: int = 30):
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.samples = deque(maxlen=window_size)

    def record(
        self,
        detection_ms: float,
        tracking_ms: float,
        analytics_ms: float,
    ) -> PerformanceSnapshot:
        pipeline_ms = (
            detection_ms + tracking_ms + analytics_ms
        )

        self.samples.append(pipeline_ms)

        mean_ms = sum(self.samples) / len(self.samples)

        return PerformanceSnapshot(
            detection_ms=detection_ms,
            tracking_ms=tracking_ms,
            analytics_ms=analytics_ms,
            pipeline_ms=pipeline_ms,
            pipeline_fps=(
                1000.0 / mean_ms if mean_ms > 0 else 0.0
            ),
        )


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
