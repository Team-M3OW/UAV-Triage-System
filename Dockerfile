FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

FROM base AS pathfinding
RUN pip install --no-cache-dir rasterio osgeo

FROM base AS trauma
RUN pip install --no-cache-dir transformers

FROM base AS detection

EXPOSE 8000

CMD ["python3", "-c", "from src.detection import alltracker; print('UAV Triage System ready')"]
