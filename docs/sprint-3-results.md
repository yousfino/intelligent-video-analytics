# Sprint 3 — Model Training, Evaluation & Benchmarking

**Project:** Intelligent Video Analytics  
**Status:** Core Sprint 3A–3D implementation complete  
**Evaluation date:** November 19, 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB)  
**Tracking:** MLflow, SQLite backend (`sqlite:///mlflow.db`)

## Executive summary

Sprint 3 delivered a deterministic pedestrian dataset split, fine-tuning of YOLO11n and RF-DETR Nano, MLflow experiment tracking, held-out test evaluation, and sequential GPU inference benchmarking.

**Key result:** RF-DETR Nano reported 84.46% mAP@50:95 and 59.25 FPS, versus YOLO11n's 74.45% mAP@50:95 and 22.03 FPS. **These are native-framework, native-configuration results—not a controlled equal-resolution, equal-precision comparison.** No definitive architecture winner is claimed.

## Sprint deliverables

| Phase     | Work                                                              | Status      |
| --------- | ----------------------------------------------------------------- | ----------- |
| 3A        | YOLO training smoke test and MLflow integration                   | Complete    |
| 3B        | Penn-Fudan dataset conversion and deterministic split             | Complete    |
| 3C        | YOLO11n and RF-DETR Nano fine-tuning; training/validation metrics | Complete    |
| 3D        | Held-out evaluation, GPU benchmarks, MLflow benchmark runs        | Complete    |
| Follow-up | Harmonized COCO evaluation and controlled performance comparison  | Outstanding |

## Dataset

**Penn-Fudan Pedestrian:** 170 images; deterministic split using seed 42.

| Split      | Images |
| ---------- | -----: |
| Train      |    118 |
| Validation |     25 |
| Test       |     27 |

The YOLO test evaluator reported **67 annotated person instances**. YOLO and COCO test image filenames matched exactly for all 27 test images. Equivalence of the bounding-box annotations has **not** yet been independently verified.

Generated dataset paths:

- `data/processed/penn_fudan/yolo/data.yaml`
- `data/processed/penn_fudan/yolo/test/images/`
- `data/processed/penn_fudan/coco/test/_annotations.coco.json`

## Model training

| Setting             | YOLO11n                                            | RF-DETR Nano                                                   |
| ------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| Base model          | `yolo11n.pt`                                       | `RFDETRNano()`                                                 |
| Epochs              | 20                                                 | 10                                                             |
| Dataset format      | YOLO                                               | COCO                                                           |
| Batch size          | 4                                                  | 1, gradient accumulation 4                                     |
| Random seed         | 42                                                 | 42                                                             |
| Selected checkpoint | `outputs/training/yolo-penn-fudan/weights/best.pt` | `outputs/training/rfdetr-penn-fudan/checkpoint_best_total.pth` |
| Checkpoint size     | 5.24 MiB                                           | 115.30 MiB                                                     |

Training and validation metrics were logged/backfilled in MLflow. **Training losses are architecture-specific and should not be compared numerically across models.**

## Held-out test evaluation

Both models were evaluated on their respective formats of the same 27-image test split.

| Metric    |    YOLO11n | RF-DETR Nano |
| --------- | ---------: | -----------: |
| Precision |     85.67% |   **93.65%** |
| Recall    | **94.03%** |       88.06% |
| mAP@50    | **95.24%** |       94.36% |
| mAP@50:95 |     74.45% |   **84.46%** |

RF-DETR additionally reported mAP@75 = 93.27%, mAR = 91.34%, and F1 = 90.77%. Its mAP@50:95 was **10.00 percentage points higher** than YOLO's reported value. YOLO reported higher recall. These scores use different framework evaluation implementations, so differences in thresholds, interpolation, and preprocessing may affect comparability.

**Evaluation configuration caveat:** YOLO evaluated at 640 × 640. RF-DETR's test evaluator logged a 384-pixel square resize. A common COCO evaluation pipeline is needed before making definitive accuracy claims.

### Reproduce evaluation

```powershell
python scripts/evaluate_yolo.py `
  --weights "outputs/training/yolo-penn-fudan/weights/best.pt" `
  --data "data/processed/penn_fudan/yolo/data.yaml"

python scripts/evaluate_rfdetr.py `
  --weights "outputs/training/rfdetr-penn-fudan/checkpoint_best_total.pth" `
  --dataset "data/processed/penn_fudan/coco"
```

Results are saved to `outputs/evaluation/yolo/metrics.json` and `outputs/evaluation/rfdetr/metrics.json`.

## GPU inference benchmarking

### Protocol

- RTX 4050 Laptop GPU; 27 test images preloaded into RAM
- Sequential single-image predictions, batch size 1
- 20 warm-up iterations, then 100 timed iterations per model
- CUDA synchronized before and after each timed prediction
- Includes API preprocessing, inference, and postprocessing; **excludes disk reads**
- FPS = `1000 / mean_latency_ms`; not measured webcam throughput
- Confidence threshold = 0.5
- GPU memory = **peak PyTorch CUDA allocated memory**, not total VRAM usage

**Different native configurations:** YOLO used `imgsz=640`, `half=False` (FP32 requested). RF-DETR used checkpoint-default inference settings; the benchmark did not explicitly capture its actual resolution or precision. RF-DETR also warned that its inference path was not optimized.

| Metric                        |       YOLO11n | RF-DETR Nano |
| ----------------------------- | ------------: | -----------: |
| Median latency                |      43.42 ms | **15.41 ms** |
| Mean latency                  |      45.39 ms | **16.88 ms** |
| P95 latency                   |      58.33 ms | **31.41 ms** |
| Sequential FPS                |         22.03 |    **59.25** |
| Peak PyTorch allocated memory | **65.97 MiB** |   147.65 MiB |
| Checkpoint size               |  **5.24 MiB** |   115.30 MiB |

In this setup, RF-DETR achieved **2.69× sequential throughput** with approximately **2.24× peak PyTorch allocated memory**. Its saved checkpoint was approximately **22× larger**; checkpoint size is not equivalent to exported inference artifact size or runtime VRAM requirements.

### Reproduce benchmarks

```powershell
python scripts/benchmark_models.py `
  --model yolo `
  --weights "outputs/training/yolo-penn-fudan/weights/best.pt" `
  --images "data/processed/penn_fudan/yolo/test/images" `
  --warmup 20 `
  --iterations 100

python scripts/benchmark_models.py `
  --model rfdetr `
  --weights "outputs/training/rfdetr-penn-fudan/checkpoint_best_total.pth" `
  --images "data/processed/penn_fudan/coco/test" `
  --warmup 20 `
  --iterations 100
```

Results are saved to `outputs/benchmark/yolo_benchmark.json` and `outputs/benchmark/rfdetr_benchmark.json`. Run the models separately under consistent power and GPU load conditions.

## MLflow results

**Experiment:** `intelligent-video-analytics`  
**Backend:** local SQLite, `mlflow.db`

| Run                                | Run ID                             |
| ---------------------------------- | ---------------------------------- |
| `yolo-penn-fudan-test-benchmark`   | `2964266115054980ab3126a46c82f75f` |
| `rfdetr-penn-fudan-test-benchmark` | `de54a91347fc468880816546625c4029` |

Each run records standardized `test/` and `benchmark/` metrics, parameters, and JSON artifacts.

```powershell
# Creates new runs each time it is executed; avoid unnecessary duplicates.
python scripts/log_benchmarks_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

View the experiment at `http://127.0.0.1:5000`. Run IDs refer to the original local MLflow database; they will not be resolvable in a fresh clone without that database.

## Limitations and follow-up work

1. Verify that YOLO and COCO annotations encode identical bounding boxes and category mappings.
2. Convert both models' detections into COCO results and score with a **single COCO evaluator**.
3. Compare inference at explicitly documented, supported input resolutions and precision settings; retain native-deployment benchmarks separately.
4. Measure forward-only latency separately from end-to-end prediction latency.
5. Record total GPU VRAM usage in addition to PyTorch peak allocated memory.
6. Repeat independent benchmark runs to quantify variability and thermal/power effects.
7. Evaluate on a larger, more varied pedestrian test set and test sustained webcam throughput, tracking, and restricted-zone event detection.

## Conclusion

Sprint 3's core training, evaluation, benchmarking, and MLflow deliverables are complete. **RF-DETR Nano is a promising GPU inference candidate** based on native-framework test mAP@50:95 and measured sequential throughput; **YOLO11n remains attractive for lightweight deployment** due to its small checkpoint, lower measured PyTorch allocation, and higher reported recall.

**Deployment selection remains provisional** until accuracy and inference are compared under a harmonized protocol.
