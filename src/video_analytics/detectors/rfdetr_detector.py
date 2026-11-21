
import cv2
import numpy as np
from rfdetr import RFDETRNano

from .base import BaseDetector, DetectionResult


class RFDETRDetector(BaseDetector):

    def __init__(
        self,
        weights: str,
        confidence: float = 0.10,
    ):
        self.model = RFDETRNano.from_checkpoint(
            weights
        )
        self.confidence = confidence

    def detect(
        self, frame: np.ndarray
    ) -> DetectionResult:

        # OpenCV BGR -> RF-DETR RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        detections = self.model.predict(
            rgb_frame,
            threshold=self.confidence,
            include_source_image=False,
        )

        if len(detections) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4)),
                scores=np.empty((0,)),
                class_ids=np.empty((0,)),
            )

        # Penn-Fudan fine-tuned model: person only.
        # Use a common class ID for downstream tracking.
        return DetectionResult(
            boxes=detections.xyxy,
            scores=detections.confidence,
            class_ids=np.zeros(
                len(detections), dtype=np.int32
            ),
        )
