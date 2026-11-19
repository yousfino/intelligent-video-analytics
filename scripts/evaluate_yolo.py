
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.val(
        data=args.data,
        split="test",
        imgsz=640,
        batch=1,
        device=args.device,
        plots=True,
        verbose=True,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    output = Path("outputs/evaluation/yolo")
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nYOLO Test Results")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
