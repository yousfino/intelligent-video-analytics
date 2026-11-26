
import logging
import threading
import time

from video_analytics.api.state import WorkerStatus, worker_state
from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector
from video_analytics.events import EventLogger
from video_analytics.pipeline import AnalyticsPipeline
from video_analytics.stream import VideoStream
from video_analytics.tracker import ByteTrackTracker
from video_analytics.zones import RestrictedZone
import os

from video_analytics.database.repository import (
    create_event_repository,
)

from video_analytics.database.writer import AsyncEventWriter

from video_analytics.api.prometheus_metrics import (
    update_worker_metrics
)

from contextlib import closing

from video_analytics.database.health import (
    evaluate_writer_health,
)

logger = logging.getLogger(__name__)


class AnalyticsWorker:
    def __init__(
        self,
        source: str | int,
        model: str = "yolo",
        loop_file: bool = False,
        config_path: str = "configs/pipeline.yaml",
    ):
        if model not in ("yolo", "rfdetr"):
            raise ValueError("Unsupported detector model")

        self.source = source
        self.model = model
        self.loop_file = loop_file
        self.config_path = config_path

        self.stop_event = threading.Event()
        self.thread = None
        self.stream = None
        self.event_writer = None

        self.database_enabled = (
            os.getenv(
                "ANALYTICS_DATABASE_ENABLED",
                "false",
            ).lower() == "true"
        )

    def database_health(self) -> dict:
        writer = self.event_writer

        snapshot = (
            writer.snapshot()
            if writer is not None
            else None
        )

        return evaluate_writer_health(
            snapshot=snapshot,
            enabled=self.database_enabled,
        )

    def start(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Worker is already running")

        self.stop_event.clear()
        self.event_writer = None

        worker_state.update(
            status=WorkerStatus.STARTING,
            model=self.model,
            source=str(self.source),
            frames_processed=0,
            intrusion_events=0,
            throughput_fps=0.0,
            error=None,
            last_frame_monotonic=None,
            stream_connected=False,
            reconnect_attempts=0,
            total_reconnections=0,
        )

        self.thread = threading.Thread(
            target=self._run,
            name="analytics-worker",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            worker_state.update(
                status=WorkerStatus.STOPPING
            )

        if self.stream is not None:
            self.stream.stop()

        if self.thread:
            self.thread.join(timeout=15)

        if self.thread and self.thread.is_alive():
            logger.warning(
                "Worker did not stop within 15 seconds"
            )
            return False

        return True


    def _on_stream_state_change(
        self,
        state: str,
        attempts: int,
    ):
        if self.stop_event.is_set():
            return

        if state == "RECONNECTING":
            worker_state.update(
                status=WorkerStatus.RECONNECTING,
                stream_connected=False,
                reconnect_attempts=attempts,
                last_frame_monotonic=None,
            )

        elif state == "CONNECTED":
            snapshot = worker_state.snapshot()

            total_reconnections = snapshot[
                "total_reconnections"
            ]

            if snapshot["frames_processed"] > 0:
                total_reconnections += 1

            worker_state.update(
                stream_connected=True,
                reconnect_attempts=0,
                total_reconnections=total_reconnections,
                error=None,
            )

        elif state == "STOPPED":
            worker_state.update(
                stream_connected=False,
                last_frame_monotonic=None,
            )


    def _run(self):
        frames_processed = 0
        intrusion_events = 0
        started_at = time.perf_counter()

        try:
            config = load_config(self.config_path)
            config["detector"]["model"] = self.model

            detector = create_detector(config)

            tracking_config = config["tracking"]

            tracker = ByteTrackTracker(
                frame_rate=tracking_config["frame_rate"],
                track_activation_threshold=tracking_config[
                    "track_activation_threshold"
                ],
                lost_track_buffer=tracking_config[
                    "lost_track_buffer"
                ],
                minimum_matching_threshold=tracking_config[
                    "minimum_matching_threshold"
                ],
            )

            zone = RestrictedZone.from_file(
                config["zones"]["config_path"]
            )

            event_logger = EventLogger(
                config["events"]["output_path"]
            )

            database_enabled = self.database_enabled

            event_writer = None

            if database_enabled:
                try:
                    event_repository = create_event_repository()

                    event_writer = AsyncEventWriter(
                        repository=event_repository,
                        max_queue_size=1000,
                        max_attempts=3,
                        retry_delay=0.5,
                        on_close=event_repository.engine.dispose,
                    )

                    event_writer.start()
                    self.event_writer = event_writer

                    logger.info(
                        "Asynchronous PostgreSQL persistence enabled"
                    )

                except Exception:
                    logger.exception(
                        "Unable to initialize database writer"
                    )
                    if event_writer is not None:
                        event_writer.stop()
                    elif "event_repository" in locals():
                        event_repository.engine.dispose()
                    event_writer = None
                    self.event_writer = None

            pipeline = AnalyticsPipeline(
                detector=detector,
                tracker=tracker,
                zone=zone,
                logger=event_logger,
                model_name=self.model,
            )

            if self.stop_event.is_set():
                return

            reconnect_delay = float(
                os.getenv(
                    "ANALYTICS_RECONNECT_DELAY_SECONDS",
                    "3",
                )
            )

            max_retries_value = int(
                os.getenv(
                    "ANALYTICS_MAX_RECONNECT_ATTEMPTS",
                    "0",
                )
            )

            if max_retries_value < 0:
                raise ValueError(
                    "Maximum reconnect attempts must be >= 0"
                )

            max_retries = (
                None
                if max_retries_value == 0
                else max_retries_value
            )

            self.stream = VideoStream(
                source=self.source,
                reconnect_delay=reconnect_delay,
                max_retries=max_retries,
                loop_file=self.loop_file,
                on_state_change=self._on_stream_state_change,
            )

            if self.stop_event.is_set():
                self.stream.stop()
                return

            with closing(self.stream.frames()) as frames:
                for frame in frames:
                    if self.stop_event.is_set():
                        break

                    frames_processed += 1

                    result = pipeline.process_frame(
                        frame,
                        frames_processed,
                    )

                    intrusion_events += len(result.events)

                    if event_writer is not None:
                        for event in result.events:
                            event_writer.submit(event)

                    elapsed = (
                        time.perf_counter() - started_at
                    )

                    throughput = (
                        frames_processed / elapsed
                        if elapsed > 0 else 0.0
                    )

                    worker_state.update(
                        status=(
                            WorkerStatus.STOPPING
                            if self.stop_event.is_set()
                            else WorkerStatus.RUNNING
                        ),
                        frames_processed=frames_processed,
                        intrusion_events=intrusion_events,
                        throughput_fps=round(throughput, 2),
                        last_frame_monotonic=time.monotonic(),
                    )

                    update_worker_metrics(
                        frames=frames_processed,
                        intrusions=intrusion_events,
                        fps=throughput,
                    )

                    if frames_processed % 100 == 0:
                        logger.info(
                            "Frames=%s FPS=%.2f Events=%s",
                            frames_processed,
                            throughput,
                            intrusion_events,
                        )

        except Exception as exc:
            logger.exception(
                "Analytics worker failed"
            )

            worker_state.update(
                status=WorkerStatus.ERROR,
                error=str(exc),
            )

        finally:

            if "event_writer" in locals() and event_writer is not None:
                stopped = event_writer.stop(timeout=15.0)

                logger.info(
                    "Database writer stopped=%s stats=%s",
                    stopped,
                    event_writer.snapshot(),
                )

                if not stopped:
                    logger.warning(
                        "Database writer did not finish draining"
                    )

            if self.stream is not None:
                self.stream.stop()

            worker_state.update(
                stream_connected=False,
                last_frame_monotonic=None,
            )

            if (
                worker_state.snapshot()["status"]
                != WorkerStatus.ERROR
            ):
                worker_state.update(
                    status=WorkerStatus.STOPPED
                )
