
import argparse
import time

import cv2

from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector
from video_analytics.tracker import ByteTrackTracker


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

    detector = create_detector(config)

    tracking_config = config.get("tracking", {})

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

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            if tracks.tracker_id is not None:
                for box, track_id in zip(
                    tracks.xyxy, tracks.tracker_id
                ):
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Person ID: {track_id}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            now = time.perf_counter()
            fps = 1.0 / max(
                now - previous_time, 1e-6
            )
            previous_time = now

            active_tracks = len(tracks)

            cv2.putText(
                frame,
                f"Model: {config['detector']['model']} "
                f"| FPS: {fps:.1f} "
                f"| Tracks: {active_tracks}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Intelligent Video Analytics - ByteTrack",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
