
import argparse
import json
import time
from pathlib import Path

import cv2

from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector
from video_analytics.events import EventLogger
from video_analytics.pipeline import AnalyticsPipeline
from video_analytics.tracker import ByteTrackTracker
from video_analytics.zones import RestrictedZone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless intelligent video analytics"
    )
    parser.add_argument(
        "--config", default="configs/pipeline.yaml"
    )
    parser.add_argument(
        "--source", required=True,
        help="Video file path or camera index"
    )
    parser.add_argument(
        "--model", choices=["yolo", "rfdetr"],
        default=None
    )
    parser.add_argument(
        "--weights", default=None,
        help="Override selected detector checkpoint path"
    )
    parser.add_argument(
        "--events", default=None,
        help="Override JSONL event output path"
    )
    parser.add_argument(
        "--summary", default=None,
        help="Optional JSON processing summary path"
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

    source = int(args.source) if args.source.isdecimal() else args.source

    detector = create_detector(config)

    tracking_config = config["tracking"]

    tracker = ByteTrackTracker(
        frame_rate=tracking_config["frame_rate"],
        track_activation_threshold=tracking_config[
            "track_activation_threshold"
        ],
        lost_track_buffer=tracking_config["lost_track_buffer"],
        minimum_matching_threshold=tracking_config[
            "minimum_matching_threshold"
        ],
    )

    zone = RestrictedZone.from_file(
        config["zones"]["config_path"]
    )

    logger = EventLogger(
        config["events"]["output_path"]
    )

    pipeline = AnalyticsPipeline(
        detector=detector,
        tracker=tracker,
        zone=zone,
        logger=logger,
        model_name=model_name,
    )

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    frame_number = 0
    total_events = 0
    started_at = time.perf_counter()
    last_performance = None

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            result = pipeline.process_frame(
                frame, frame_number
            )

            total_events += len(result.events)
            last_performance = result.performance

            for event in result.events:
                print(
                    "INTRUSION:",
                    json.dumps(event),
                    flush=True,
                )

            if frame_number % 30 == 0:
                elapsed = time.perf_counter() - started_at
                fps = frame_number / elapsed if elapsed > 0 else 0

                print(
                    f"Frames: {frame_number} | "
                    f"Processing throughput: {fps:.2f} FPS | "
                    f"Events: {total_events}",
                    flush=True,
                )

    finally:
        capture.release()

    elapsed = time.perf_counter() - started_at

    summary = {
        "model": model_name,
        "source": str(source),
        "frames_processed": frame_number,
        "intrusion_events": total_events,
        "elapsed_seconds": round(elapsed, 3),
        "average_throughput_fps": round(
            frame_number / elapsed, 2
        ) if elapsed > 0 else 0,
        "events_path": config["events"]["output_path"],
    }

    if last_performance is not None:
        summary["rolling_pipeline_fps"] = (
            last_performance.pipeline_fps
        )

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    print("\nProcessing completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
