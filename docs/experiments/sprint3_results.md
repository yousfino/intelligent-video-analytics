# Sprint 3 — Model Fine-Tuning

## Hardware

- GPU: NVIDIA RTX 4050 Laptop (6 GB)
- Framework: PyTorch + CUDA
- Experiment tracking: MLflow + SQLite

## Dataset

- Name: Penn-Fudan Pedestrian
- Train: 118 images
- Validation: 25 images
- Test: 27 images
- Split seed: 42
- Classes: person

## Model 1 — YOLO11n

- Pretrained weights: yolo11n.pt
- Epochs: 20
- Batch size: 4
- Image size: 640
- Validation mAP@0.50:
- Validation mAP@0.50:0.95:
- Checkpoint:

## Model 2 — RF-DETR Nano

- Pretrained weights: RF-DETR Nano
- Epochs: 10
- Batch size: 1
- Gradient accumulation: 4
- Validation mAP@0.50:
- Validation mAP@0.50:0.95:
- Checkpoint:

## Observations

- Training stability:
- GPU memory limitations:
- Dataset limitations:
- Potential improvements:

## Next

Evaluate both models on the held-out test set
using a consistent evaluation protocol.
