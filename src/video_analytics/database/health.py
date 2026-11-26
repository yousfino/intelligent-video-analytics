
import math
import os


def evaluate_writer_health(
    snapshot: dict | None,
    enabled: bool,
    max_heartbeat_age: float | None = None,
    max_consecutive_failures: int | None = None,
) -> dict:
    if not enabled:
        return {
            "enabled": False,
            "healthy": None,
            "reasons": [],
            "stats": None,
        }

    if max_heartbeat_age is None:
        max_heartbeat_age = float(
            os.getenv(
                "ANALYTICS_DB_WRITER_TIMEOUT_SECONDS",
                "30",
            )
        )

    if max_consecutive_failures is None:
        max_consecutive_failures = int(
            os.getenv(
                "ANALYTICS_DB_MAX_CONSECUTIVE_FAILURES",
                "3",
            )
        )

    if (
        not math.isfinite(max_heartbeat_age)
        or max_heartbeat_age <= 0
    ):
        raise ValueError(
            "Database writer timeout must be positive"
        )

    if max_consecutive_failures < 1:
        raise ValueError(
            "Maximum consecutive failures must be positive"
        )

    reasons = []

    if snapshot is None:
        reasons.append("Database writer unavailable")

    else:
        if not snapshot["alive"]:
            reasons.append("Database writer thread is not alive")

        if snapshot["stopping"]:
            reasons.append("Database writer is stopping")

        if snapshot["fatal_error"] is not None:
            reasons.append("Database writer terminated unexpectedly")

        heartbeat_age = snapshot["heartbeat_age_seconds"]

        if (
            heartbeat_age is None
            or heartbeat_age > max_heartbeat_age
        ):
            reasons.append("Database writer heartbeat is stale")

        if snapshot["queue_utilization"] >= 1.0:
            reasons.append("Database event queue is full")

        if (
            snapshot["consecutive_failures"]
            >= max_consecutive_failures
        ):
            reasons.append(
                "Database writer failure threshold exceeded"
            )

    return {
        "enabled": True,
        "healthy": not reasons,
        "reasons": reasons,
        "stats": snapshot,
    }
