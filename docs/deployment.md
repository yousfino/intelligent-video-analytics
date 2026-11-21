# Intelligent Video Analytics — Deployment Guide

## Prerequisites

- Linux server or Windows development PC with Docker Desktop and WSL2
- NVIDIA GPU compatible with the selected CUDA/PyTorch environment
- NVIDIA driver compatible with CUDA 12.6
- Docker Engine with Docker Compose
- NVIDIA Container Toolkit on native Linux
- Sufficient disk space for the CUDA-enabled image, model checkpoints, and video files

## Application Components

- YOLO11n or RF-DETR object detector
- ByteTrack multi-object tracking
- Restricted-zone intrusion detection
- JSONL event logging
- JSON performance summaries

## Directory Structure

```text
intelligent-video-analytics/
├── compose.prod.yaml
├── inputs/
│   └── test_video.mp4
├── outputs/
│   ├── training/
│   ├── events/
│   └── headless/
└── configs/
    ├── pipeline.yaml
    └── zones.json
```

## Deployment

Build the GPU image on a compatible machine:

```bash
docker build -f Dockerfile.gpu -t intelligent-video-analytics:gpu .
```

Validate the deployment configuration:

```bash
docker compose -f compose.prod.yaml config --quiet
```

Start video processing:

```bash
ANALYTICS_MODEL=yolo docker compose -f compose.prod.yaml up --abort-on-container-exit --exit-code-from analytics
```

For RF-DETR:

```bash
ANALYTICS_MODEL=rfdetr docker compose -f compose.prod.yaml up --force-recreate --abort-on-container-exit --exit-code-from analytics
```

## Persistence

The following directories are mounted from the host:

- `inputs/`: Input videos
- `outputs/`: Checkpoints, JSONL events, and processing summaries
- `configs/`: Application configuration

Removing containers does not delete files stored in these directories.

## Current Limitations

- Processes finite video files rather than continuously monitoring cameras
- Requires an NVIDIA GPU for the GPU image
- No remote management API or dashboard
- No container registry publishing workflow yet
- No automated production deployment or rollback
- No production health checks or centralized monitoring

## Validation

A successful deployment must:

1. Detect the NVIDIA GPU.
2. Load the selected model checkpoint.
3. Process the complete input video.
4. Write the processing summary.
5. Persist intrusion events when detected.
6. Exit with code 0.
7. Preserve outputs after container removal.