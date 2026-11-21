
from pathlib import Path

import yaml


def load_config(
    path: str | Path = "configs/pipeline.yaml",
) -> dict:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Pipeline configuration must be a YAML mapping"
        )

    if "detector" not in config:
        raise ValueError(
            "Missing detector configuration"
        )

    return config
