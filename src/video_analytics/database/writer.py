
import logging
import math
import queue
import threading
import time

from typing import Any, Callable


logger = logging.getLogger(__name__)


class AsyncEventWriter:
    def __init__(
        self,
        repository,
        max_queue_size: int = 1000,
        max_attempts: int = 3,
        retry_delay: float = 0.5,
        on_close: Callable[[], None] | None = None,
    ):
        if max_queue_size < 1 or max_attempts < 1:
            raise ValueError(
                "Queue size and attempts must be positive"
            )

        if not math.isfinite(retry_delay) or retry_delay < 0:
            raise ValueError(
                "Retry delay must be finite and non-negative"
            )

        self.repository = repository
        self.events = queue.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.on_close = on_close

        self.stop_event = threading.Event()
        self.thread = None

        self._lock = threading.Lock()

        self.saved = 0
        self.failed = 0
        self.dropped = 0
        self.consecutive_failures = 0

        self.last_heartbeat = None
        self.last_success = None
        self.last_error = None
        self.fatal_error = None
        self.in_flight = False

    def start(self):
        with self._lock:
            if self.thread is not None:
                raise RuntimeError("Writer already started")

            self.stop_event.clear()
            self.last_heartbeat = time.monotonic()

            self.thread = threading.Thread(
                target=self._run,
                name="database-event-writer",
                daemon=True,
            )

            self.thread.start()

    def submit(self, event: dict[str, Any]) -> bool:
        with self._lock:
            if self.thread is None:
                return False

            if (
                self.stop_event.is_set()
                or not self.thread.is_alive()
            ):
                self.dropped += 1

                logger.error(
                    "Database writer unavailable; event dropped"
                )

                return False

            try:
                self.events.put_nowait(dict(event))
                return True

            except queue.Full:
                self.dropped += 1

                logger.error(
                    "Database event queue is full; event dropped"
                )

                return False

    def _heartbeat(self):
        with self._lock:
            self.last_heartbeat = time.monotonic()

    def _run(self):
        try:
            while True:
                self._heartbeat()

                if (
                    self.stop_event.is_set()
                    and self.events.empty()
                ):
                    break

                try:
                    event = self.events.get(timeout=0.1)

                except queue.Empty:
                    continue

                with self._lock:
                    self.in_flight = True

                try:
                    success = False

                    for attempt in range(self.max_attempts):
                        self._heartbeat()

                        try:
                            self.repository.save(event)

                            success = True
                            break

                        except Exception as exc:
                            with self._lock:
                                self.last_error = str(exc)

                            logger.exception(
                                "Database write failed "
                                "(attempt %s/%s)",
                                attempt + 1,
                                self.max_attempts,
                            )

                            if attempt + 1 < self.max_attempts:
                                time.sleep(self.retry_delay)

                    with self._lock:
                        if success:
                            self.saved += 1
                            self.consecutive_failures = 0
                            self.last_success = time.monotonic()
                            self.last_error = None

                        else:
                            self.failed += 1
                            self.consecutive_failures += 1

                finally:
                    with self._lock:
                        self.in_flight = False
                        self.last_heartbeat = time.monotonic()

                    self.events.task_done()

        except BaseException as exc:
            with self._lock:
                self.fatal_error = str(exc) or type(exc).__name__
                self.last_error = self.fatal_error
                self.failed += 1
                self.consecutive_failures += 1

            logger.exception(
                "Database writer terminated unexpectedly"
            )

        finally:
            if self.on_close is not None:
                try:
                    self.on_close()

                except Exception:
                    logger.exception(
                        "Database resource cleanup failed"
                    )

    def stop(self, timeout: float = 15.0) -> bool:
        with self._lock:
            self.stop_event.set()
            thread = self.thread

        if thread is None:
            return True

        thread.join(timeout=timeout)

        return not thread.is_alive()

    def snapshot(self):
        with self._lock:
            now = time.monotonic()

            alive = (
                self.thread is not None
                and self.thread.is_alive()
            )

            pending = self.events.qsize()

            heartbeat_age = (
                max(0.0, now - self.last_heartbeat)
                if self.last_heartbeat is not None
                else None
            )

            return {
                "saved": self.saved,
                "failed": self.failed,
                "dropped": self.dropped,
                "pending": pending,
                "alive": alive,
                "stopping": self.stop_event.is_set(),
                "in_flight": self.in_flight,
                "queue_capacity": self.max_queue_size,
                "queue_utilization": (
                    pending / self.max_queue_size
                ),
                "heartbeat_age_seconds": heartbeat_age,
                "last_success_age_seconds": (
                    max(0.0, now - self.last_success)
                    if self.last_success is not None
                    else None
                ),
                "consecutive_failures": (
                    self.consecutive_failures
                ),
                "last_error": self.last_error,
                "fatal_error": self.fatal_error,
            }
