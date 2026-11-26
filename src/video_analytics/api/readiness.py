
import logging
import math
import os
import time

from video_analytics.api.state import (
    WorkerStatus,
    worker_state,
)

from video_analytics.database.connection import (
    check_database_connection,
)


logger = logging.getLogger(__name__)


def evaluate_readiness(
    database_health: dict | None = None,
) -> dict:
    snapshot = worker_state.snapshot()

    checks = {}
    reasons = []

    # Worker lifecycle status
    worker_running = (
        snapshot["status"] == WorkerStatus.RUNNING
    )

    checks["worker_running"] = worker_running

    stream_connected = snapshot["stream_connected"]

    checks["stream_connected"] = stream_connected

    if not stream_connected:
        reasons.append("Video source disconnected")

    if not worker_running:
        reasons.append(
            f"Worker status: {snapshot['status'].value}"
        )

    # Frame processing heartbeat
    max_frame_age = float(
        os.getenv(
            "ANALYTICS_MAX_FRAME_AGE_SECONDS",
            "15",
        )
    )

    if not math.isfinite(max_frame_age) or max_frame_age <= 0:
        raise ValueError(
            "ANALYTICS_MAX_FRAME_AGE_SECONDS must be positive"
        )

    last_frame = snapshot["last_frame_monotonic"]

    frame_age = (
        max(0.0, time.monotonic() - last_frame)
        if last_frame is not None
        else None
    )

    frame_processing = (
        frame_age is not None
        and frame_age <= max_frame_age
    )

    checks["frame_processing"] = frame_processing

    if last_frame is None:
        reasons.append("No frames processed yet")

    elif not frame_processing:
        reasons.append("Frame processing heartbeat is stale")

    # Optional database dependency
    database_enabled = (
        os.getenv(
            "ANALYTICS_DATABASE_ENABLED",
            "false",
        ).lower() == "true"
    )

    database_available = None

    if database_enabled:
        try:
            check_database_connection()
            database_available = True

        except Exception:
            logger.warning(
                "Readiness database check failed",
                exc_info=True,
            )

            database_available = False
            reasons.append("PostgreSQL unavailable")

    checks["database_available"] = database_available
    writer_healthy = None

    if database_enabled:
        writer_healthy = (
            database_health is not None
            and database_health["healthy"] is True
        )

        if not writer_healthy:
            if database_health is None:
                reasons.append("Database writer unavailable")
            else:
                reasons.extend(database_health["reasons"])

    checks["database_writer_healthy"] = writer_healthy

    ready = (
        worker_running
        and stream_connected
        and frame_processing
        and (
            not database_enabled
            or (
                database_available is True
                and writer_healthy is True
            )
        )
    )

    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "worker_status": snapshot["status"],
        "frames_processed": snapshot["frames_processed"],
        "last_frame_age_seconds": (
            round(frame_age, 3)
            if frame_age is not None
            else None
        ),
        "database_required": database_enabled,
        "reasons": reasons,
    }
