
import logging
import threading
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoStream:
    def __init__(
        self,
        source: str | int,
        reconnect_delay: float = 3.0,
        max_retries: int | None = None,
        loop_file: bool = False,
    ):
        if reconnect_delay < 0:
            raise ValueError("reconnect_delay must be >= 0")
        if max_retries is not None and max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        self.source = source
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.loop_file = loop_file

        self.capture = None
        self.running = False
        self._stop_event = threading.Event()

    @property
    def is_file(self) -> bool:
        if not isinstance(self.source, str):
            return False
        if "://" in self.source:
            return False
        return Path(self.source).is_file()

    def open(self) -> bool:
        self.close()
        self.capture = cv2.VideoCapture(self.source)

        if self.capture.isOpened():
            logger.info("Video source connected: %s", self.source)
            return True

        self.close()
        return False

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def stop(self):
        self.running = False
        self._stop_event.set()
        self.close()

    def frames(self) -> Iterator[np.ndarray]:
        self._stop_event.clear()
        self.running = True
        retries = 0

        try:
            while not self._stop_event.is_set():
                if self.capture is None:
                    if not self.open():
                        retries += 1
                        logger.warning(
                            "Connection failed (%s/%s)",
                            retries,
                            self.max_retries or "unlimited",
                        )

                        if (
                            self.max_retries is not None
                            and retries >= self.max_retries
                        ):
                            raise RuntimeError(
                                "Maximum connection retries exceeded"
                            )

                        if self._stop_event.wait(
                            self.reconnect_delay
                        ):
                            break

                        continue

                    retries = 0

                success, frame = self.capture.read()

                if success:
                    yield frame
                    continue

                if self.is_file:
                    if self.loop_file:
                        self.capture.set(
                            cv2.CAP_PROP_POS_FRAMES, 0
                        )
                        continue
                    break

                logger.warning(
                    "Video source disconnected: %s",
                    self.source,
                )

                self.close()

                if self._stop_event.wait(
                    self.reconnect_delay
                ):
                    break

        finally:
            self.stop()
