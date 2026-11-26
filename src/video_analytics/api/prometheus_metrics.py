
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

db_writer_enabled = Gauge(
    "iva_db_writer_enabled",
    "Whether database persistence is enabled",
    registry=registry,
)

db_writer_alive = Gauge(
    "iva_db_writer_alive",
    "Whether the database writer thread is alive",
    registry=registry,
)

db_writer_healthy = Gauge(
    "iva_db_writer_healthy",
    "Whether the database writer is healthy",
    registry=registry,
)

db_writer_heartbeat_age = Gauge(
    "iva_db_writer_heartbeat_age_seconds",
    "Seconds since the last database writer heartbeat",
    registry=registry,
)

db_queue_utilization = Gauge(
    "iva_db_queue_utilization",
    "Fraction of database queue capacity in use",
    registry=registry,
)

db_consecutive_failures = Gauge(
    "iva_db_consecutive_failures",
    "Consecutive events that failed persistence",
    registry=registry,
)

db_write_in_flight = Gauge(
    "iva_db_write_in_flight",
    "Whether a database event is being processed",
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



def update_database_health_metrics(health: dict):
    snapshot = health["stats"] or {}

    db_writer_enabled.set(
        int(health["enabled"])
    )

    db_writer_alive.set(
        int(snapshot.get("alive", False))
    )

    db_writer_healthy.set(
        int(health["healthy"] is True)
    )

    heartbeat_age = snapshot.get(
        "heartbeat_age_seconds"
    )

    db_writer_heartbeat_age.set(
        heartbeat_age
        if heartbeat_age is not None
        else float("nan")
    )

    db_queue_utilization.set(
        snapshot.get("queue_utilization", 0)
    )

    db_consecutive_failures.set(
        snapshot.get("consecutive_failures", 0)
    )

    db_write_in_flight.set(
        int(snapshot.get("in_flight", False))
    )


def render_metrics():
    return generate_latest(registry)
