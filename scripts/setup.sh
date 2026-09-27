#!/usr/bin/env bash
# One-time setup: MediaMTX binary, YOLO weights, Python environment.
set -euo pipefail
cd "$(dirname "$0")/.."

MEDIAMTX_VERSION=${MEDIAMTX_VERSION:-v1.21.1}
FACE_MODEL_URL=https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11s-face.pt
OBJECT_MODEL_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt

command -v ffmpeg >/dev/null || { echo "ffmpeg is required (apt install ffmpeg)"; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
nvidia-smi >/dev/null 2>&1 || echo "WARNING: nvidia-smi not found - NVENC/CUDA are required"

mkdir -p bin models
if [[ ! -x bin/mediamtx ]] || ! bin/mediamtx --version 2>/dev/null | grep -q "$MEDIAMTX_VERSION"; then
  echo "downloading mediamtx $MEDIAMTX_VERSION"
  curl -fsSL "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/mediamtx_${MEDIAMTX_VERSION}_linux_amd64.tar.gz" \
    | tar xz -C bin mediamtx
fi
[[ -f models/yolov11s-face.pt ]] || curl -fL -o models/yolov11s-face.pt "$FACE_MODEL_URL"
[[ -f models/yolo11s.pt ]] || curl -fL -o models/yolo11s.pt "$OBJECT_MODEL_URL"

uv sync --frozen
echo
echo "done. start with:  uv run live-masker"
