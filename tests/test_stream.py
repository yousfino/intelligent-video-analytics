
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


def test_stream_recovers_after_disconnect():
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    first_capture = MagicMock()
    first_capture.isOpened.return_value = True
    first_capture.read.side_effect = [
        (True, frame),
        (False, None),
    ]

    second_capture = MagicMock()
    second_capture.isOpened.return_value = True
    second_capture.read.return_value = (True, frame)

    callback = MagicMock()

    with patch(
        "video_analytics.stream.cv2.VideoCapture",
        side_effect=[
            first_capture,
            second_capture,
        ],
    ):
        stream = VideoStream(
            source=0,
            reconnect_delay=0,
            on_state_change=callback,
        )

        generator = stream.frames()

        next(generator)
        next(generator)

        generator.close()

    callback.assert_any_call("CONNECTED", 0)
    callback.assert_any_call("RECONNECTING", 1)
    callback.assert_any_call("CONNECTED", 1)
    callback.assert_any_call("STOPPED", 0)

    first_capture.release.assert_called_once()
    second_capture.release.assert_called_once()


def test_retry_limit_for_failed_reads():
    callback = MagicMock()

    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (False, None)

    with patch(
        "video_analytics.stream.cv2.VideoCapture",
        return_value=capture,
    ):
        stream = VideoStream(
            source=0,
            reconnect_delay=0,
            max_retries=3,
            on_state_change=callback,
        )

        with pytest.raises(
            RuntimeError,
            match="Maximum connection retries exceeded",
        ):
            list(stream.frames())

    assert capture.release.call_count == 3

    callback.assert_any_call("RECONNECTING", 1)
    callback.assert_any_call("RECONNECTING", 2)
    callback.assert_any_call("RECONNECTING", 3)


def test_normal_file_loop_is_not_reconnection(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.set.return_value = True

    capture.read.side_effect = [
        (True, frame),
        (False, None),
        (True, frame),
    ]

    callback = MagicMock()

    with patch(
        "video_analytics.stream.cv2.VideoCapture",
        return_value=capture,
    ):
        stream = VideoStream(
            source=str(video_path),
            loop_file=True,
            reconnect_delay=0,
            on_state_change=callback,
        )

        generator = stream.frames()

        next(generator)
        next(generator)

        generator.close()

    states = [
        call.args[0]
        for call in callback.call_args_list
    ]

    assert states == ["CONNECTED", "STOPPED"]

    capture.set.assert_called_once()


def test_stream_stops_during_reconnection():
    capture = MagicMock()
    capture.isOpened.return_value = False

    def callback(state, attempts):
        if state == "RECONNECTING":
            stream.stop()

    with patch(
        "video_analytics.stream.cv2.VideoCapture",
        return_value=capture,
    ):
        stream = VideoStream(
            source=0,
            reconnect_delay=60,
            on_state_change=callback,
        )

        assert list(stream.frames()) == []

    assert stream.running is False
    capture.release.assert_called_once()


def test_stopped_stream_does_not_reopen():
    stream = VideoStream(source=0)

    stream.stop()

    with patch(
        "video_analytics.stream.cv2.VideoCapture"
    ) as video_capture:
        assert list(stream.frames()) == []

        video_capture.assert_not_called()
