
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRNano
from ultralytics import YOLO


def load_images(directory):
    extensions = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise RuntimeError("No test images found")

    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(np.array(img.convert("RGB")))

    return images


def benchmark(name, predict_fn, images, warmup, iterations):
    print(f"\nBenchmarking {name}...")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Warm-up: initialize CUDA kernels and model execution paths.
    for i in range(warmup):
        predict_fn(images[i % len(images)])

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timings = []

    for i in range(iterations):
        image = images[i % len(images)]

        torch.cuda.synchronize()
        start = time.perf_counter()

        predict_fn(image)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        timings.append(elapsed_ms)

    median_ms = statistics.median(timings)
    mean_ms = statistics.mean(timings)
    p95_ms = float(np.percentile(timings, 95))

    results = {
        "model": name,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "median_latency_ms": median_ms,
        "mean_latency_ms": mean_ms,
        "p95_latency_ms": p95_ms,
        "fps_sequential": 1000.0 / mean_ms,
        "peak_pytorch_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
        ),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["yolo", "rfdetr"],
        required=True,
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    images = load_images(args.images)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Images loaded:", len(images))
    print("Model:", args.model)

    if args.model == "yolo":
        model = YOLO(args.weights)

        def predict(image):
            return model.predict(
                source=image[:, :, ::-1].copy(),
                imgsz=640,
                device=0,
                half=False,
                conf=0.5,
                verbose=False,
                save=False,
            )

    else:
        model = RFDETRNano.from_checkpoint(
            str(Path(args.weights).resolve())
        )

        def predict(image):
            return model.predict(
                image,
                threshold=0.5,
                include_source_image=False,
            )

    # Ensure model is initialized before measuring.
    predict(images[0])

    results = benchmark(
        args.model,
        predict,
        images,
        args.warmup,
        args.iterations,
    )

    results["checkpoint_size_mb"] = (
        Path(args.weights).stat().st_size / 1024**2
    )
    results["gpu"] = torch.cuda.get_device_name(0)
    results["input_resolution"] = (
        "640x640" if args.model == "yolo"
        else "checkpoint_default"
    )
    results["precision"] = (
        "FP32" if args.model == "yolo"
        else "RF-DETR checkpoint default"
    )
    results["confidence_threshold"] = 0.5

    output_dir = Path("outputs/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.model}_benchmark.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nBENCHMARK RESULTS")
    print(json.dumps(results, indent=4))
    print("\nSaved to:", output_file)


if __name__ == "__main__":
    main()
