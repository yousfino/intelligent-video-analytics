
import argparse
import csv
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path.as_posix()}"
    )

    client = MlflowClient()
    run = client.get_run(args.run_id)

    print("Updating run:", run.info.run_name)
    print("Run ID:", run.info.run_id)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Training CSV is empty")

    metric_mapping = {
        "train/box_loss": "train/box_loss",
        "train/cls_loss": "train/cls_loss",
        "train/dfl_loss": "train/dfl_loss",
        "val/box_loss": "val/box_loss",
        "val/cls_loss": "val/cls_loss",
        "val/dfl_loss": "val/dfl_loss",
        "metrics/precision(B)": "val/precision",
        "metrics/recall(B)": "val/recall",
        "metrics/mAP50(B)": "val/mAP50",
        "metrics/mAP50-95(B)": "val/mAP50_95",
    }

    for row in rows:
        epoch = int(float(row["epoch"]))

        for csv_key, mlflow_key in metric_mapping.items():
            if csv_key not in row:
                continue

            value = row[csv_key].strip()
            if not value:
                continue

            client.log_metric(
                run.info.run_id,
                mlflow_key,
                float(value),
                step=epoch,
            )

    print(f"Logged {len(rows)} epochs successfully")


if __name__ == "__main__":
    main()
