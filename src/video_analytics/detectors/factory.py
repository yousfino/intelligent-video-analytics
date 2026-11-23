
from pathlib import Path

from .base import BaseDetector
from .yolo_detector import YOLODetector
from .rfdetr_detector import RFDETRDetector


def create_detector(config: dict) -> BaseDetector:
    detector_config = config["detector"]

    model = detector_config["model"].lower()
    confidence = float(
        detector_config.get("confidence", 0.10)
    )

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Detector confidence must be between 0 and 1"
        )

    if model == "yolo":
        settings = detector_config["yolo"]
        weights = Path(settings["weights"])

        if not weights.is_file():
            raise FileNotFoundError(weights)

        return YOLODetector(
            weights=str(weights),
            confidence=confidence,
            device=settings.get("device", 0),
            image_size=int(
                settings.get("image_size", 640)
            ),
        )

    if model == "rfdetr":
        settings = detector_config["rfdetr"]
        weights = Path(settings["weights"])

        if not weights.is_file():
            raise FileNotFoundError(weights)

        return RFDETRDetector(
            weights=str(weights),
            confidence=confidence,
        )

    raise ValueError(
        f"Unsupported detector model: {model}"
    )
