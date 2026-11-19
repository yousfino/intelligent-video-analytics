
import argparse
import csv
import math
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    mlflow.set_tracking_uri(
        f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
    )

    client = MlflowClient()
    run = client.get_run(args.run_id)

    print("Updating run:", run.info.run_name)
    print("Run ID:", run.info.run_id)

    count = 0
    epochs = set()

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epoch = int(float(row["epoch"]))
            epochs.add(epoch)

            for name, raw_value in row.items():
                if not name or name in ("epoch", "step"):
                    continue

                if not name.startswith(("train/", "val/")):
                    continue

                if raw_value is None or not raw_value.strip():
                    continue

                value = float(raw_value)

                if not math.isfinite(value):
                    continue

                client.log_metric(
                    run.info.run_id,
                    name,
                    value,
                    step=epoch,
                )
                count += 1

    print(f"Imported {count} metric values")
    print(f"Epochs found: {len(epochs)}")
    print("Done!")


if __name__ == "__main__":
    main()
