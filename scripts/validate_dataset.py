
import json
from pathlib import Path


def validate(root: Path):
    manifest = json.loads(
        (root / "split_manifest.json").read_text(encoding="utf-8")
    )

    all_images = []

    for split, names in manifest["splits"].items():
        coco_file = root / "coco" / split / "_annotations.coco.json"
        coco = json.loads(coco_file.read_text(encoding="utf-8"))

        assert len(names) == len(coco["images"])
        assert len(names) == len(set(names))

        coco_images = {im["id"]: im for im in coco["images"]}
        coco_counts = {image_id: 0 for image_id in coco_images}

        for ann in coco["annotations"]:
            assert ann["image_id"] in coco_images
            assert ann["category_id"] == 1
            assert ann["area"] > 0
            coco_counts[ann["image_id"]] += 1

        for image in coco["images"]:
            filename = image["file_name"]
            stem = Path(filename).stem

            assert (root / "coco" / split / filename).exists()
            assert (root / "yolo" / split / "images" / filename).exists()

            label_path = root / "yolo" / split / "labels" / f"{stem}.txt"
            assert label_path.exists()

            lines = label_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == coco_counts[image["id"]]

            for line in lines:
                values = line.split()
                assert len(values) == 5
                assert values[0] == "0"

                xc, yc, w, h = map(float, values[1:])
                assert 0 <= xc <= 1 and 0 <= yc <= 1
                assert 0 < w <= 1 and 0 < h <= 1
                assert xc - w / 2 >= -1e-6
                assert xc + w / 2 <= 1 + 1e-6
                assert yc - h / 2 >= -1e-6
                assert yc + h / 2 <= 1 + 1e-6

        all_images.extend(names)
        print(f"{split}: validated {len(names)} images")

    assert len(all_images) == len(set(all_images)), (
        "Dataset leakage: image appears in multiple splits"
    )

    print("Dataset validation PASSED")


if __name__ == "__main__":
    validate(Path("data/processed/penn_fudan"))
