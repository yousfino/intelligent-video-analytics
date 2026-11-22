
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from video_analytics.stream import VideoStream


def test_invalid_reconnect_delay():
    with pytest.raises(ValueError):
        VideoStream(0, reconnect_delay=-1)


def test_invalid_retry_limit():
    with pytest.raises(ValueError):
        VideoStream(0, max_retries=0)


def test_successful_frame_capture():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)

    with patch(
        "video_analytics.stream.cv2.VideoCapture",
        return_value=capture,
    ):
        stream = VideoStream(0)

        generator = stream.frames()
        result = next(generator)

        assert result.shape == (480, 640, 3)

        generator.close()
        assert capture.release.called


def test_maximum_retries():
    capture = MagicMock()
    capture.isOpened.return_value = False

    with patch(
        "video_analytics.stream.cv2.VideoCapture",
        return_value=capture,
    ):
        stream = VideoStream(
            source=0,
            reconnect_delay=0,
            max_retries=3,
        )

        with pytest.raises(
            RuntimeError,
            match="Maximum connection retries exceeded",
        ):
            list(stream.frames())

        assert capture.release.call_count == 3


def test_stop_releases_capture():
    capture = MagicMock()

    stream = VideoStream(0)
    stream.capture = capture

    stream.stop()

    capture.release.assert_called_once()
    assert stream.capture is None
    assert stream.running is False
