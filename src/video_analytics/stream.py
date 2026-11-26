
import logging
import math
import threading

from collections.abc import Callable, Iterator
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
        on_state_change: Callable[[str, int], None] | None = None,
    ):
        if (
            not math.isfinite(reconnect_delay)
            or reconnect_delay < 0
        ):
            raise ValueError(
                "reconnect_delay must be finite and >= 0"
            )

        if max_retries is not None and max_retries < 1:
            raise ValueError(
                "max_retries must be >= 1"
            )

        self.source = source
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.loop_file = loop_file
        self.on_state_change = on_state_change

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

    def _notify(self, state: str, attempts: int = 0):
        if self.on_state_change is not None:
            self.on_state_change(state, attempts)

    def open(self) -> bool:
        self.close()

        self.capture = cv2.VideoCapture(self.source)

        if self.capture.isOpened():
            logger.info(
                "Video capture opened: %s",
                self.source,
            )

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

        # The frame generator owns the capture while active.
        # Avoid releasing it concurrently with capture.read().
        if not getattr(self, "_iterating", False):
            self.close()

    def _retry(self, attempts: int) -> bool:
        self.close()

        self._notify("RECONNECTING", attempts)

        logger.warning(
            "Video source retry (%s/%s)",
            attempts,
            self.max_retries or "unlimited",
        )

        if (
            self.max_retries is not None
            and attempts >= self.max_retries
        ):
            raise RuntimeError(
                "Maximum connection retries exceeded"
            )

        return not self._stop_event.wait(
            self.reconnect_delay
        )

    def frames(self) -> Iterator[np.ndarray]:
        if self._stop_event.is_set():
            return

        self.running = True
        self._iterating = True

        attempts = 0
        connected = False
        rewound = False

        try:
            while not self._stop_event.is_set():
                if self.capture is None:
                    if not self.open():
                        attempts += 1

                        if not self._retry(attempts):
                            break

                        continue

                if self._stop_event.is_set():
                    break

                success, frame = self.capture.read()

                if self._stop_event.is_set():
                    break

                if success and frame is not None:
                    rewound = False

                    if not connected:
                        self._notify("CONNECTED", attempts)

                        logger.info(
                            "Video frames available"
                        )

                    connected = True
                    attempts = 0

                    yield frame
                    continue

                if self.is_file:
                    if not self.loop_file:
                        break

                    if not rewound:
                        rewound = True

                        if self.capture.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            0,
                        ):
                            continue

                connected = False
                rewound = False
                attempts += 1

                logger.warning(
                    "Video source interrupted"
                )

                if not self._retry(attempts):
                    break

        finally:
            self.running = False
            self._stop_event.set()
            self.close()
            self._iterating = False

            self._notify("STOPPED")
