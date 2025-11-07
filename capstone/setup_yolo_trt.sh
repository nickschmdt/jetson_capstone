#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
APP_DIR="$HOME/capstone"
VENV_DIR="$APP_DIR/.venv"
ENGINE_PATH="$APP_DIR/yolo11n_fp16.engine"
ONNX_PATH="$APP_DIR/yolo11n.onnx"
MODEL_PT="yolo11n.pt"
IMG_SIZE=640
TRT_MEM="workspace:4096M"

# --- Sanity: groups & devices ---
id | grep -E '\b(video|render)\b' >/dev/null || {
  echo ">>> Add your user to 'video' and 'render' groups and reboot:"
  echo "sudo usermod -aG video,render $USER"
  exit 1
}
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# --- Python venv ---
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

# 1) Remove pip OpenCV (prevents NumPy 2.x wheel loops)
pip uninstall -y opencv-python opencv-contrib-python || true

# 2) Pin NumPy <2 and a few basics
pip install --upgrade "pip<25"
pip install --upgrade "numpy<2" pillow pyyaml requests "packaging<24.2"

# 3) Use system OpenCV (GStreamer-enabled)
sudo apt-get update
sudo apt-get install -y python3-opencv
python - << 'PY'
import cv2, numpy as np
print("OK OpenCV:", cv2.__version__, "NumPy:", np.__version__)
PY

# 4) Ultralytics in the venv
pip install --upgrade ultralytics

# 5) CUDA-enabled PyTorch matching JetPack 6.x
pip uninstall -y torch torchvision || true
pip install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# (Optional) tolerate missing torchvision in Ultralytics
python - << 'PY'
from importlib import metadata
import pathlib, ultralytics
p = pathlib.Path(ultralytics.__file__).resolve().parent / 'utils' / '__init__.py'
s = p.read_text()
needle = 'TORCHVISION_VERSION = importlib.metadata.version("torchvision")'
if needle in s and 'PackageNotFoundError' not in s:
    s = s.replace(needle,
                  'try:\n    TORCHVISION_VERSION = importlib.metadata.version("torchvision")\n'
                  'except importlib.metadata.PackageNotFoundError:\n    TORCHVISION_VERSION = None')
    p.write_text(s)
    print("Patched Ultralytics to tolerate missing torchvision:", p)
else:
    print("Ultralytics already OK or patch not needed.")
PY

# 6) Export ONNX if missing
if [ ! -f "$ONNX_PATH" ]; then
  echo ">>> Exporting ONNX..."
  yolo export model="$MODEL_PT" format=onnx imgsz=$IMG_SIZE dynamic=False opset=12 simplify=True device=cpu
  test -f "$ONNX_PATH" || { echo "ONNX export did not produce $ONNX_PATH"; exit 1; }
fi

# 7) Build FP16 TensorRT engine (TRT 10.x syntax)
if [ ! -f "$ENGINE_PATH" ]; then
  echo ">>> Building TensorRT engine..."
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --memPoolSize="$TRT_MEM"
fi

# 8) Sanity: load CUDA driver & deserialize engine
python - << PY
import os, ctypes
os.environ["PYTHONNOUSERSITE"]="1"
os.environ["LD_LIBRARY_PATH"]="/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH","")
ctypes.CDLL("libcuda.so.1")
import tensorrt as trt; print("TRT:", trt.__version__)
from ultralytics.nn.autobackend import AutoBackend
import torch
ab = AutoBackend("$ENGINE_PATH", device=torch.device("cpu"))
print("TensorRT engine deserialized OK")
PY

echo ">>> Setup complete."
