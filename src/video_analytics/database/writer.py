
import logging
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
            raise ValueError("Queue size and attempts must be positive")
        if retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")

        self.repository = repository
        self.events = queue.Queue(maxsize=max_queue_size)
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.on_close = on_close

        self.stop_event = threading.Event()
        self.thread = None

        self._lock = threading.Lock()
        self.saved = 0
        self.failed = 0
        self.dropped = 0

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Writer already started")

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="database-event-writer",
            daemon=True,
        )
        self.thread.start()

    def submit(self, event: dict[str, Any]) -> bool:
        if self.thread is None or self.stop_event.is_set():
            return False

        try:
            self.events.put_nowait(dict(event))
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            logger.error("Database event queue is full; event dropped")
            return False

    def _run(self):
        try:
            while not self.stop_event.is_set() or not self.events.empty():
                try:
                    event = self.events.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    success = False

                    for attempt in range(self.max_attempts):
                        try:
                            self.repository.save(event)
                            success = True
                            break
                        except Exception:
                            logger.exception(
                                "Database write failed (attempt %s/%s)",
                                attempt + 1,
                                self.max_attempts,
                            )

                            if attempt + 1 < self.max_attempts:
                                time.sleep(self.retry_delay)

                    with self._lock:
                        if success:
                            self.saved += 1
                        else:
                            self.failed += 1

                finally:
                    self.events.task_done()
        finally:
            if self.on_close is not None:
                try:
                    self.on_close()
                except Exception:
                    logger.exception("Database resource cleanup failed")

    def stop(self, timeout: float = 15.0) -> bool:
        self.stop_event.set()

        if self.thread is None:
            return True

        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def snapshot(self):
        with self._lock:
            return {
                "saved": self.saved,
                "failed": self.failed,
                "dropped": self.dropped,
                "pending": self.events.qsize(),
            }
