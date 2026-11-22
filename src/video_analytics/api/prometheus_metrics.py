
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

registry = CollectorRegistry()

frames_processed = Gauge(
    "iva_frames_processed",
    "Frames processed by the current analytics worker",
    registry=registry,
)

intrusion_events = Gauge(
    "iva_intrusion_events",
    "Intrusion events detected by the current worker",
    registry=registry,
)

throughput_fps = Gauge(
    "iva_throughput_fps",
    "Average inference throughput in frames per second",
    registry=registry,
)

db_events_saved = Gauge(
    "iva_db_events_saved",
    "Database events saved by the current writer",
    registry=registry,
)

db_events_failed = Gauge(
    "iva_db_events_failed",
    "Database events that exhausted all retry attempts",
    registry=registry,
)

db_events_dropped = Gauge(
    "iva_db_events_dropped",
    "Events dropped because the database queue was full",
    registry=registry,
)

db_queue_depth = Gauge(
    "iva_db_queue_depth",
    "Current number of events waiting in the database queue",
    registry=registry,
)


def update_worker_metrics(
    frames: int,
    intrusions: int,
    fps: float,
):
    frames_processed.set(frames)
    intrusion_events.set(intrusions)
    throughput_fps.set(fps)


def update_database_metrics(snapshot: dict):
    db_events_saved.set(snapshot["saved"])
    db_events_failed.set(snapshot["failed"])
    db_events_dropped.set(snapshot["dropped"])
    db_queue_depth.set(snapshot["pending"])


def render_metrics():
    return generate_latest(registry)
