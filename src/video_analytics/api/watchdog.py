
import logging
import math
import os
import threading
import time

from collections.abc import Callable

from video_analytics.api.state import (
    WorkerStatus,
    worker_state,
)


logger = logging.getLogger(__name__)


class WorkerWatchdog:
    def __init__(
        self,
        check_interval: float = 5.0,
        startup_timeout: float = 120.0,
        frame_timeout: float = 60.0,
        reconnect_timeout: float = 0.0,
        snapshot_provider: Callable = worker_state.snapshot,
        on_failure: Callable | None = None,
        clock: Callable = time.monotonic,
    ):
        settings = {
            "check_interval": check_interval,
            "startup_timeout": startup_timeout,
            "frame_timeout": frame_timeout,
        }

        for name, value in settings.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if (
            not math.isfinite(reconnect_timeout)
            or reconnect_timeout < 0
        ):
            raise ValueError(
                "reconnect_timeout must be finite and >= 0"
            )

        self.check_interval = check_interval
        self.startup_timeout = startup_timeout
        self.frame_timeout = frame_timeout
        self.reconnect_timeout = reconnect_timeout

        self.snapshot_provider = snapshot_provider
        self.on_failure = (
            on_failure
            if on_failure is not None
            else self._terminate_process
        )

        self.clock = clock
        self.stop_event = threading.Event()
        self.thread = None

        self.started_at = self.clock()
        self.reconnecting_since = None
        self.failure_reported = False

    @staticmethod
    def _terminate_process(reason: str):
        logger.critical(
            "Watchdog terminating API process: %s",
            reason,
        )

        logging.shutdown()

        os._exit(1)

    def start(self):
        if self.thread is not None:
            raise RuntimeError(
                "Watchdog already started"
            )

        self.started_at = self.clock()
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            name="worker-watchdog",
            daemon=True,
        )

        self.thread.start()

        logger.info("Worker watchdog started")

    def stop(self, timeout: float = 5.0) -> bool:
        self.stop_event.set()

        if self.thread is None:
            return True

        self.thread.join(timeout=timeout)

        return not self.thread.is_alive()

    def evaluate(self) -> str | None:
        snapshot = self.snapshot_provider()
        status = snapshot["status"]
        now = self.clock()

        if status != WorkerStatus.RECONNECTING:
            self.reconnecting_since = None

        if status == WorkerStatus.STOPPING:
            return None

        if status == WorkerStatus.ERROR:
            return (
                "Analytics worker entered ERROR: "
                f"{snapshot.get('error')}"
            )

        if status == WorkerStatus.STOPPED:
            return "Analytics worker stopped unexpectedly"

        if status == WorkerStatus.STARTING:
            elapsed = now - self.started_at

            if elapsed > self.startup_timeout:
                return "Analytics worker startup timed out"

            return None

        if status == WorkerStatus.RECONNECTING:
            if self.reconnecting_since is None:
                self.reconnecting_since = now

            elapsed = now - self.reconnecting_since

            if (
                self.reconnect_timeout > 0
                and elapsed > self.reconnect_timeout
            ):
                return "Video reconnection timed out"

            return None

        if status == WorkerStatus.RUNNING:
            last_frame = snapshot.get(
                "last_frame_monotonic"
            )

            if last_frame is None:
                return "Running worker has no frame heartbeat"

            if now - last_frame > self.frame_timeout:
                return "Frame processing heartbeat timed out"

        return None

    def check_once(self):
        if (
            self.stop_event.is_set()
            or self.failure_reported
        ):
            return

        reason = self.evaluate()

        if (
            reason is not None
            and not self.stop_event.is_set()
        ):
            self.failure_reported = True

            logger.error(
                "Watchdog detected failure: %s",
                reason,
            )

            self.on_failure(reason)

    def _run(self):
        while not self.stop_event.wait(
            self.check_interval
        ):
            self.check_once()

            if self.failure_reported:
                break
