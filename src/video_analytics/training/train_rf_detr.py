
import argparse
import json
import re
from pathlib import Path

import mlflow
import torch
from rfdetr import RFDETRNano


def clean_metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_./ -]", "_", name)


def train(
    dataset: str,
    epochs: int,
    batch: int,
    grad_accum: int,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")

    dataset_path = Path(dataset).resolve()
    output_path = Path(
        "outputs/training/rfdetr-penn-fudan"
    ).resolve()

    if not (dataset_path / "train" /
            "_annotations.coco.json").exists():
        raise FileNotFoundError(
            f"COCO training annotations missing: {dataset_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("intelligent-video-analytics")

    params = {
        "model": "rfdetr-nano",
        "dataset": str(dataset_path),
        "epochs": epochs,
        "batch_size": batch,
        "grad_accum_steps": grad_accum,
        "learning_rate": 1e-4,
        "seed": 42,
    }

    with mlflow.start_run(run_name="rfdetr-penn-fudan"):
        mlflow.log_params(params)

        model = RFDETRNano()

        model.train(
            dataset_dir=str(dataset_path),
            output_dir=str(output_path),
            epochs=epochs,
            batch_size=batch,
            grad_accum_steps=grad_accum,
            eval_batch_size=1,
            lr=1e-4,
            device="cuda",
            num_workers=0,
            amp_dtype="auto",
            seed=42,
            tensorboard=False,
            mlflow=False,
        )

        # Evaluate on validation data only.
        metrics = model.evaluate(
            dataset_dir=str(dataset_path),
            split="val",
            batch_size=1,
            num_workers=0,
            device="cuda",
        )

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(
                    clean_metric_name(name),
                    float(value),
                )

        # Record checkpoint locations and training metadata.
        checkpoint_paths = sorted(
            str(p) for p in output_path.glob("*.pth")
        )

        manifest = {
            "params": params,
            "metrics": metrics,
            "checkpoints": checkpoint_paths,
        }

        manifest_path = output_path / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(manifest_path))

        best_checkpoint = (
            output_path / "checkpoint_best_total.pth"
        )

        if best_checkpoint.exists():
            mlflow.log_artifact(
                str(best_checkpoint),
                artifact_path="checkpoints",
            )

        print("\nRF-DETR training completed")
        print(f"Outputs: {output_path}")
        print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data/processed/penn_fudan/coco",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--grad-accum", type=int, default=4
    )

    args = parser.parse_args()

    train(
        dataset=args.dataset,
        epochs=args.epochs,
        batch=args.batch,
        grad_accum=args.grad_accum,
    )
