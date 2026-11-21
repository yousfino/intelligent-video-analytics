
import numpy as np
from ultralytics import YOLO

from .base import BaseDetector, DetectionResult


class YOLODetector(BaseDetector):

    def __init__(
        self,
        weights: str,
        confidence: float = 0.10,
        device: int = 0,
        image_size: int = 640,
    ):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device
        self.image_size = image_size

    def detect(
        self, frame: np.ndarray
    ) -> DetectionResult:

        results = self.model.predict(
            source=frame,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4)),
                scores=np.empty((0,)),
                class_ids=np.empty((0,)),
            )

        return DetectionResult(
            boxes=boxes.xyxy.cpu().numpy(),
            scores=boxes.conf.cpu().numpy(),
            class_ids=np.zeros(
                len(boxes), dtype=np.int32
            ),
        )
