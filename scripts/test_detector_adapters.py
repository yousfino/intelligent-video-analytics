
import argparse

import cv2

from video_analytics.detectors.yolo_detector import YOLODetector
from video_analytics.detectors.rfdetr_detector import RFDETRDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["yolo", "rfdetr"],
        required=True,
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    if args.model == "yolo":
        detector = YOLODetector(args.weights)
    else:
        detector = RFDETRDetector(args.weights)

    frame = cv2.imread(args.image)

    if frame is None:
        raise RuntimeError("Unable to load image")

    result = detector.detect(frame)

    print("Model:", args.model)
    print("Detections:", len(result.boxes))
    print("Boxes:", result.boxes)
    print("Scores:", result.scores)
    print("Class IDs:", result.class_ids)


if __name__ == "__main__":
    main()
