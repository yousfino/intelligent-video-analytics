
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock


class WorkerStatus(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass
class WorkerSnapshot:
    status: WorkerStatus = WorkerStatus.STOPPED
    model: str | None = None
    source: str | None = None
    frames_processed: int = 0
    intrusion_events: int = 0
    throughput_fps: float = 0.0
    error: str | None = None
    last_frame_monotonic: float | None = None
    stream_connected: bool = False
    reconnect_attempts: int = 0
    total_reconnections: int = 0


class WorkerState:
    def __init__(self):
        self._lock = Lock()
        self._snapshot = WorkerSnapshot()

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._snapshot, key):
                    raise ValueError(
                        f"Unknown worker field: {key}"
                    )

                setattr(self._snapshot, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return asdict(self._snapshot)


worker_state = WorkerState()
