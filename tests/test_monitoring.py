
import pytest

from video_analytics.monitoring import PerformanceMonitor


def test_performance_calculation():
    monitor = PerformanceMonitor()

    result = monitor.record(
        detection_ms=20,
        tracking_ms=5,
        analytics_ms=5,
    )

    assert result.pipeline_ms == 30
    assert result.pipeline_fps == pytest.approx(33.3333, rel=1e-3)


def test_rolling_average():
    monitor = PerformanceMonitor(window_size=2)

    monitor.record(10, 0, 0)
    result = monitor.record(30, 0, 0)

    assert result.pipeline_fps == pytest.approx(50.0)


def test_invalid_window_size():
    with pytest.raises(ValueError):
        PerformanceMonitor(window_size=0)
