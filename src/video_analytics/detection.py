
import argparse
import time

import cv2
import torch
from ultralytics import YOLO


def run_detection(source: int | str = 0) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = 0
    model = YOLO("yolo11n.pt")

    camera = cv2.VideoCapture(source)

    if not camera.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            results = model.predict(
                source=frame,
                device=device,
                imgsz=640,
                conf=0.35,
                classes=[0],  # COCO person class
                verbose=False,
            )

            annotated_frame = results[0].plot()

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                annotated_frame,
                f"Loop FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Intelligent Video Analytics", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_detection(source)
