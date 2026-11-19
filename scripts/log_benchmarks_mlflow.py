
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "yolo": {
        "run_name": "yolo-penn-fudan-test-benchmark",
        "accuracy_file": ROOT / "outputs/evaluation/yolo/metrics.json",
        "benchmark_file": ROOT / "outputs/benchmark/yolo_benchmark.json",
        "checkpoint": ROOT / "outputs/training/yolo-penn-fudan/weights/best.pt",
        "accuracy_mapping": {
            "precision": "precision",
            "recall": "recall",
            "mAP50": "mAP50",
            "mAP50_95": "mAP50_95",
        },
    },
    "rfdetr": {
        "run_name": "rfdetr-penn-fudan-test-benchmark",
        "accuracy_file": ROOT / "outputs/evaluation/rfdetr/metrics.json",
        "benchmark_file": ROOT / "outputs/benchmark/rfdetr_benchmark.json",
        "checkpoint": ROOT / "outputs/training/rfdetr-penn-fudan/checkpoint_best_total.pth",
        "accuracy_mapping": {
            "test/precision": "precision",
            "test/recall": "recall",
            "test/mAP_50": "mAP50",
            "test/mAP_50_95": "mAP50_95",
        },
    },
}


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main():
    mlflow.set_tracking_uri(
        (ROOT / "mlflow.db").resolve().as_uri().replace(
            "file:///", "sqlite:///"
        )
    )

    mlflow.set_experiment("intelligent-video-analytics")

    client = MlflowClient()

    for model_name, config in MODELS.items():
        accuracy = load_json(config["accuracy_file"])
        benchmark = load_json(config["benchmark_file"])

        with mlflow.start_run(
            run_name=config["run_name"]
        ) as run:

            mlflow.set_tags({
                "model_family": model_name,
                "dataset": "Penn-Fudan Pedestrian",
                "split": "test",
                "evaluation_type": "native_framework",
                "benchmark_type": "sequential_end_to_end",
            })

            mlflow.log_params({
                "checkpoint": str(config["checkpoint"]),
                "test_images": 27,
                "warmup_iterations": benchmark["warmup_iterations"],
                "timed_iterations": benchmark["timed_iterations"],
                "gpu": benchmark["gpu"],
                "input_resolution": benchmark["input_resolution"],
                "precision": benchmark["precision"],
                "confidence_threshold": benchmark["confidence_threshold"],
            })

            for source_key, target_key in config[
                "accuracy_mapping"
            ].items():
                mlflow.log_metric(
                    f"test/{target_key}",
                    float(accuracy[source_key]),
                )

            for key in [
                "median_latency_ms",
                "mean_latency_ms",
                "p95_latency_ms",
                "fps_sequential",
                "peak_pytorch_memory_mb",
                "checkpoint_size_mb",
            ]:
                mlflow.log_metric(
                    f"benchmark/{key}",
                    float(benchmark[key]),
                )

            mlflow.log_artifact(
                str(config["accuracy_file"]),
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                str(config["benchmark_file"]),
                artifact_path="benchmark",
            )

            print(
                f"Logged {model_name}: "
                f"{run.info.run_id}"
            )

    print("\nBoth benchmark runs logged successfully.")


if __name__ == "__main__":
    main()
