
import argparse
import json
from pathlib import Path

from rfdetr import RFDETRNano


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    dataset = Path(args.dataset).resolve()

    if not weights.is_file():
        raise FileNotFoundError(weights)

    if not (dataset / "test" / "_annotations.coco.json").is_file():
        raise FileNotFoundError(
            "COCO test annotations not found"
        )

    print("Loading RF-DETR checkpoint:", weights)

    model = RFDETRNano.from_checkpoint(str(weights))

    print("Evaluating held-out test split...")

    results = model.evaluate(
        dataset_dir=str(dataset),
        split="test",
        batch_size=1,
        num_workers=0,
        device="cuda",
    )

    metrics = {
        str(key): float(value)
        for key, value in results.items()
        if hasattr(value, "__float__")
    }

    output = Path("outputs/evaluation/rfdetr")
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nRF-DETR Test Results")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
