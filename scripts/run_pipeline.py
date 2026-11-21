
import argparse

import cv2

from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector
from video_analytics.events import EventLogger
from video_analytics.pipeline import AnalyticsPipeline
from video_analytics.tracker import ByteTrackTracker
from video_analytics.zones import RestrictedZone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/pipeline.yaml"
    )
    parser.add_argument(
        "--model", choices=["yolo", "rfdetr"]
    )
    parser.add_argument("--source")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["detector"]["model"] = args.model

    model_name = config["detector"]["model"]
    detector = create_detector(config)

    tracking_config = config["tracking"]

    tracker = ByteTrackTracker(
        frame_rate=int(
            tracking_config.get("frame_rate", 30)
        ),
        track_activation_threshold=float(
            tracking_config.get(
                "track_activation_threshold", 0.25
            )
        ),
        lost_track_buffer=int(
            tracking_config.get("lost_track_buffer", 30)
        ),
        minimum_matching_threshold=float(
            tracking_config.get(
                "minimum_matching_threshold", 0.8
            )
        ),
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

    source = (
        args.source
        if args.source is not None
        else config.get("video", {}).get("source", 0)
    )

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source}"
        )

    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            result = pipeline.process_frame(
                frame, frame_number
            )

            height, width = frame.shape[:2]
            polygon = zone.polygon(width, height)

            overlay = frame.copy()
            cv2.fillPoly(
                overlay, [polygon], (0, 0, 255)
            )

            frame = cv2.addWeighted(
                overlay, 0.15, frame, 0.85, 0
            )

            cv2.polylines(
                frame, [polygon], True,
                (0, 0, 255), 2
            )

            if result.tracks.tracker_id is not None:
                for box, track_id in zip(
                    result.tracks.xyxy,
                    result.tracks.tracker_id,
                ):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)

                    inside = track_id in result.inside_ids
                    color = (
                        (0, 0, 255)
                        if inside else (0, 255, 0)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1), (x2, y2),
                        color, 2
                    )

                    cv2.circle(
                        frame,
                        (int((x1 + x2) / 2), y2),
                        5, color, -1
                    )

                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )

            for event in result.events:
                print(f"[ALERT] {event}")

            metrics = result.performance

            cv2.putText(
                frame,
                f"{model_name.upper()} | "
                f"Pipeline FPS: "
                f"{metrics.pipeline_fps:.1f} | "
                f"Tracks: {len(result.tracks)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            status = (
                "INTRUSION DETECTED"
                if result.inside_ids
                else "MONITORING"
            )

            cv2.putText(
                frame, status, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255)
                if result.inside_ids
                else (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Detection: {metrics.detection_ms:.1f}ms "
                f"| Tracking: {metrics.tracking_ms:.1f}ms "
                f"| Analytics: {metrics.analytics_ms:.1f}ms",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            cv2.imshow(
                "Intelligent Video Analytics",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
