
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

    def start(self):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Worker is already running")

        self.stop_event.clear()

        worker_state.update(
            status=WorkerStatus.STARTING,
            model=self.model,
            source=str(self.source),
            frames_processed=0,
            intrusion_events=0,
            throughput_fps=0.0,
            error=None,
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
            
            database_enabled = (
                os.getenv("ANALYTICS_DATABASE_ENABLED", "false")
                .lower() == "true"
            )

            event_repository = None

            if database_enabled:
                try:
                    event_repository = create_event_repository()
                    logger.info("PostgreSQL event persistence enabled")
                except Exception:
                    logger.exception(
                        "Unable to initialize database repository"
                    )

            pipeline = AnalyticsPipeline(
                detector=detector,
                tracker=tracker,
                zone=zone,
                logger=event_logger,
                model_name=self.model,
            )

            if self.stop_event.is_set():
                return

            self.stream = VideoStream(
                source=self.source,
                reconnect_delay=3.0,
                max_retries=None,
                loop_file=self.loop_file,
            )

            worker_state.update(
                status=WorkerStatus.RUNNING
            )

            for frame in self.stream.frames():
                if self.stop_event.is_set():
                    break

                frames_processed += 1

                result = pipeline.process_frame(
                    frame,
                    frames_processed,
                )

                intrusion_events += len(result.events)
                
                if event_repository is not None:
                    for event in result.events:
                        try:
                            event_repository.save(event)
                        except Exception:
                            logger.exception(
                                "Failed to persist intrusion event "
                                "at frame %s",
                                frames_processed,
                            )

                elapsed = (
                    time.perf_counter() - started_at
                )

                throughput = (
                    frames_processed / elapsed
                    if elapsed > 0 else 0.0
                )

                worker_state.update(
                    frames_processed=frames_processed,
                    intrusion_events=intrusion_events,
                    throughput_fps=round(
                        throughput, 2
                    ),
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
            if self.stream is not None:
                self.stream.stop()

            if (
                worker_state.snapshot()["status"]
                != WorkerStatus.ERROR
            ):
                worker_state.update(
                    status=WorkerStatus.STOPPED
                )
