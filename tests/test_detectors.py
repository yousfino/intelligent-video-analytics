
import numpy as np
import pytest

from video_analytics.detectors.base import DetectionResult


def test_detection_result_shapes():
    result = DetectionResult(
        boxes=[[10, 20, 100, 200]],
        scores=[0.95],
        class_ids=[0],
    )

    assert result.boxes.shape == (1, 4)
    assert result.scores.shape == (1,)
    assert result.class_ids.shape == (1,)


def test_empty_detection_result():
    result = DetectionResult(
        boxes=[],
        scores=[],
        class_ids=[],
    )

    assert result.boxes.shape == (0, 4)
    assert result.scores.shape == (0,)
    assert result.class_ids.shape == (0,)


def test_mismatched_detection_lengths():
    with pytest.raises(ValueError):
        DetectionResult(
            boxes=[[0, 0, 10, 10]],
            scores=[],
            class_ids=[0],
        )
