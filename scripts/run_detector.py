
import argparse
import time

import cv2

from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
    )
    parser.add_argument(
        "--model",
        choices=["yolo", "rfdetr"],
        default=None,
    )
    parser.add_argument("--source", default=None)

    args = parser.parse_args()

    config = load_config(args.config)

    if args.model is not None:
        config["detector"]["model"] = args.model

    detector = create_detector(config)

    source = (
        args.source
        if args.source is not None
        else config.get("video", {}).get("source", 0)
    )

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source}"
        )

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            detections = detector.detect(frame)

            for box, score in zip(
                detections.boxes,
                detections.scores,
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
                    f"Person {score:.2f}",
                    (x1, max(y1 - 10, 0)),
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

            cv2.putText(
                frame,
                f"Model: {config['detector']['model']} "
                f"| FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Intelligent Video Analytics",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
