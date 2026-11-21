
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# OpenCV and common inference runtime libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .

# Install CPU PyTorch first
RUN python -m pip install --no-cache-dir \
    torch==2.14.1 \
    torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
RUN python -m pip install --no-cache-dir \
    -r requirements-runtime.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

CMD ["python", "-c", "import torch, cv2, rfdetr, supervision, ultralytics; print('Inference dependencies loaded successfully'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"]
