
import re
import argparse
from pathlib import Path

import mlflow
import torch
from ultralytics import YOLO


def train(
    dataset: str,
    epochs: int,
    batch: int,
    image_size: int,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("intelligent-video-analytics")

    with mlflow.start_run(run_name="yolo-penn-fudan"):
        params = {
            "model": "yolo11n.pt",
            "dataset": dataset,
            "epochs": epochs,
            "batch": batch,
            "image_size": image_size,
            "device": 0,
        }

        mlflow.log_params(params)

        model = YOLO("yolo11n.pt")

        results = model.train(
            data=dataset,
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            device=0,
            workers=2,
            project=str(Path("outputs/training").resolve()),
            name="yolo-penn-fudan",
            exist_ok=True,
            plots=True,
            seed=42,
            patience=10,
            amp=True,
        )

        metrics = results.results_dict

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_name = re.sub(r"[^a-zA-Z0-9_./ -]", "_", name)
                mlflow.log_metric(clean_name, float(value))

        checkpoint = Path(results.save_dir) / "weights" / "test.pt"

        if checkpoint.exists():
            mlflow.log_artifact(
                str(checkpoint),
                artifact_path="checkpoints",
            )

        print("Training complete.")
        print(f"Checkpoint: {checkpoint}")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="coco128.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    args = parser.parse_args()

    train(
        args.dataset,
        args.epochs,
        args.batch,
        args.imgsz,
    )
