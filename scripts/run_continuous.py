
import argparse
import json
import signal
import threading
import time
from pathlib import Path

from video_analytics.stream import VideoStream
from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector
from video_analytics.events import EventLogger
from video_analytics.pipeline import AnalyticsPipeline
from video_analytics.tracker import ByteTrackTracker
from video_analytics.zones import RestrictedZone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous intelligent video analytics"
    )

    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video file path, camera index, or RTSP URL",
    )
    parser.add_argument(
        "--model",
        choices=["yolo", "rfdetr"],
        default=None,
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Override selected detector checkpoint path",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Override JSONL event output path",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional JSON processing summary path",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously loop local video files",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.model:
        config["detector"]["model"] = args.model

    model_name = config["detector"]["model"]

    if args.weights:
        config["detector"][model_name]["weights"] = args.weights

    if args.events:
        config["events"]["output_path"] = args.events

    source = (
        int(args.source)
        if args.source.isdecimal()
        else args.source
    )

    # Initialize the selected detector.
    detector = create_detector(config)

    # Initialize ByteTrack.
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

    # Load restricted zones.
    zone = RestrictedZone.from_file(
        config["zones"]["config_path"]
    )

    # Initialize event logger.
    logger = EventLogger(
        config["events"]["output_path"]
    )

    # Construct analytics pipeline.
    pipeline = AnalyticsPipeline(
        detector=detector,
        tracker=tracker,
        zone=zone,
        logger=logger,
        model_name=model_name,
    )

    # Initialize continuous video stream.
    stream = VideoStream(
        source=source,
        reconnect_delay=3.0,
        max_retries=None,
        loop_file=args.loop,
    )

    stop_event = threading.Event()

    def request_shutdown(signum, frame):
        print(
            "\nShutdown requested...",
            flush=True,
        )
        stop_event.set()
        stream.stop()

    signal.signal(
        signal.SIGINT,
        request_shutdown,
    )

    signal.signal(
        signal.SIGTERM,
        request_shutdown,
    )

    frame_number = 0
    total_events = 0
    started_at = time.perf_counter()
    last_performance = None

    print(
        f"Starting continuous analytics | "
        f"Model: {model_name} | "
        f"Source: {source}",
        flush=True,
    )

    try:
        for frame in stream.frames():
            if stop_event.is_set():
                break

            frame_number += 1

            result = pipeline.process_frame(
                frame,
                frame_number,
            )

            total_events += len(result.events)
            last_performance = result.performance

            for event in result.events:
                print(
                    "INTRUSION:",
                    json.dumps(event),
                    flush=True,
                )

            if frame_number % 100 == 0:
                elapsed = (
                    time.perf_counter()
                    - started_at
                )

                fps = (
                    frame_number / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"Frames: {frame_number} | "
                    f"Throughput: {fps:.2f} FPS | "
                    f"Events: {total_events}",
                    flush=True,
                )

    except KeyboardInterrupt:
        # Fallback if Ctrl+C interrupts a blocking call.
        stop_event.set()
        print(
            "\nKeyboard interrupt received.",
            flush=True,
        )

    finally:
        stream.stop()

        elapsed = (
            time.perf_counter()
            - started_at
        )

        summary = {
            "model": model_name,
            "source": str(source),
            "frames_processed": frame_number,
            "intrusion_events": total_events,
            "elapsed_seconds": round(
                elapsed, 3
            ),
            "average_throughput_fps": round(
                frame_number / elapsed, 2
            ) if elapsed > 0 else 0,
            "events_path": config[
                "events"
            ]["output_path"],
        }

        if last_performance is not None:
            summary[
                "rolling_pipeline_fps"
            ] = last_performance.pipeline_fps

        if args.summary:
            summary_path = Path(
                args.summary
            )

            summary_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            summary_path.write_text(
                json.dumps(
                    summary,
                    indent=2,
                ),
                encoding="utf-8",
            )

        print(
            "\nContinuous worker stopped.",
            flush=True,
        )

        print(
            json.dumps(
                summary,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
