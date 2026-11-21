
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    def __post_init__(self):
        self.boxes = np.asarray(
            self.boxes, dtype=np.float32
        ).reshape(-1, 4)

        self.scores = np.asarray(
            self.scores, dtype=np.float32
        ).reshape(-1)

        self.class_ids = np.asarray(
            self.class_ids, dtype=np.int32
        ).reshape(-1)

        count = len(self.boxes)

        if not (
            len(self.scores) == count
            and len(self.class_ids) == count
        ):
            raise ValueError(
                "Boxes, scores, and class IDs must align"
            )


class BaseDetector(ABC):

    @abstractmethod
    def detect(
        self, frame: np.ndarray
    ) -> DetectionResult:
        """Detect objects in a BGR OpenCV frame."""
        raise NotImplementedError
