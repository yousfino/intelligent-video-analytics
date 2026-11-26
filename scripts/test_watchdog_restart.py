
import logging
import time

from pathlib import Path

from video_analytics.api.state import (
    WorkerStatus,
    worker_state,
)

from video_analytics.api.watchdog import WorkerWatchdog


logging.basicConfig(level=logging.INFO)

marker = Path("/tmp/watchdog-restart-marker")

watchdog = WorkerWatchdog(
    check_interval=2.0,
    startup_timeout=30.0,
    frame_timeout=5.0,
)

if not marker.exists():
    marker.touch()

    print(
        "First execution: simulating stalled inference",
        flush=True,
    )

    worker_state.update(
        status=WorkerStatus.RUNNING,
        last_frame_monotonic=time.monotonic(),
    )

    watchdog.start()

    while True:
        time.sleep(1)

else:
    print(
        "Second execution: simulating healthy inference",
        flush=True,
    )

    worker_state.update(
        status=WorkerStatus.RUNNING,
        last_frame_monotonic=time.monotonic(),
    )

    watchdog.start()

    try:
        while True:
            worker_state.update(
                last_frame_monotonic=time.monotonic(),
            )

            time.sleep(1)

    finally:
        watchdog.stop()
