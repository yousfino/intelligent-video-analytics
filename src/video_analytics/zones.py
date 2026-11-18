
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RestrictedZone:
    name: str
    normalized_points: list[list[float]]
    active_ids: set[int] = field(default_factory=set)

    @classmethod
    def from_file(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        points = config["points"]

        if len(points) < 3 or any(
            len(p) != 2 or
            any(not isinstance(v, (int, float)) or
                not 0 <= v <= 1 for v in p)
            for p in points
        ):
            raise ValueError("Invalid normalized polygon")

        return cls(
            name=config["name"],
            normalized_points=points,
        )

    def polygon(self, width: int, height: int):
        points = np.array(
            self.normalized_points,
            dtype=np.float32,
        )

        points[:, 0] *= width
        points[:, 1] *= height

        return np.rint(points).astype(np.int32)

    def contains(
        self,
        point: tuple[int, int],
        width: int,
        height: int,
    ) -> bool:
        polygon = self.polygon(width, height)

        return cv2.pointPolygonTest(
            polygon,
            (float(point[0]), float(point[1])),
            False,
        ) >= 0

    def update(self, inside_ids: set[int]) -> set[int]:
        """Return IDs newly entering the restricted zone."""
        entered = inside_ids - self.active_ids
        self.active_ids = inside_ids.copy()
        return entered
