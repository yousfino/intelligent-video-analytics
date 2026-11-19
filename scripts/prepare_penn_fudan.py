
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(raw_dir: Path, output_dir: Path, seed: int):
    image_dir = raw_dir / "PNGImages"
    mask_dir = raw_dir / "PedMasks"

    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    # Deterministic split, shared by YOLO and COCO.
    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_valid],
        "test": images[n_train + n_valid:],
    }

    manifest = {
        "dataset": "PennFudanPed",
        "seed": seed,
        "splits": {
            name: [p.name for p in paths]
            for name, paths in splits.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    for split_name, paths in splits.items():
        yolo_images = output_dir / "yolo" / split_name / "images"
        yolo_labels = output_dir / "yolo" / split_name / "labels"
        coco_dir = output_dir / "coco" / split_name

        for directory in (yolo_images, yolo_labels, coco_dir):
            directory.mkdir(parents=True, exist_ok=True)

        coco = {
            "info": {
                "description": "Penn-Fudan pedestrian detection",
                "version": "1.0",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                }
            ],
        }

        annotation_id = 1

        for image_id, image_path in enumerate(paths, start=1):
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            with Image.open(image_path) as im:
                width, height = im.size

            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image)

            if mask.ndim != 2:
                raise ValueError(f"Expected 2D instance mask: {mask_path}")

            if mask.shape != (height, width):
                raise ValueError(f"Mask/image size mismatch: {image_path}")

            shutil.copy2(image_path, yolo_images / image_path.name)
            shutil.copy2(image_path, coco_dir / image_path.name)

            coco["images"].append({
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            })

            yolo_lines = []

            for instance_id in np.unique(mask):
                if instance_id == 0:
                    continue  # Background

                ys, xs = np.where(mask == instance_id)
                if len(xs) == 0:
                    continue

                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1

                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    continue

                # YOLO: class, normalized center x/y, width/height
                xc = (x1 + x2) / (2 * width)
                yc = (y1 + y2) / (2 * height)
                bw = box_width / width
                bh = box_height / height

                yolo_lines.append(
                    f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
                )

                # COCO: top-left x/y, pixel width/height
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                })
                annotation_id += 1

            (yolo_labels / f"{image_path.stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )

        (coco_dir / "_annotations.coco.json").write_text(
            json.dumps(coco, indent=2),
            encoding="utf-8",
        )

        print(
            f"{split_name}: {len(paths)} images, "
            f"{len(coco['annotations'])} pedestrian boxes"
        )

    # Absolute dataset root avoids Ultralytics resolving
    # relative paths against a different datasets directory.
    yolo_root = (output_dir / "yolo").resolve().as_posix()

    yaml_content = (
        f"path: '{yolo_root}'\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: person\n"
    )

    (output_dir / "yolo" / "data.yaml").write_text(
        yaml_content,
        encoding="utf-8",
    )

    print(f"\nDataset prepared at: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        default="data/raw/PennFudanPed",
    )
    parser.add_argument(
        "--output",
        default="data/processed/penn_fudan",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare(
        Path(args.raw),
        Path(args.output),
        args.seed,
    )
