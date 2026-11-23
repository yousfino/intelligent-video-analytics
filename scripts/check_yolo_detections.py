
import cv2

from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector

config = load_config("configs/pipeline.docker-cpu.yaml")
detector = create_detector(config)

capture = cv2.VideoCapture("inputs/test_video.mp4")

frames = 0
frames_with_detections = 0
total_detections = 0

try:
    while frames < 100:
        ok, frame = capture.read()

        if not ok:
            break

        frames += 1
        result = detector.detect(frame)

        count = len(result.boxes)
        total_detections += count

        if count > 0:
            frames_with_detections += 1
finally:
    capture.release()

print(f"Frames inspected: {frames}")
print(f"Frames with detections: {frames_with_detections}")
print(f"Total person detections: {total_detections}")
