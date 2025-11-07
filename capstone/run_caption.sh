#!/usr/bin/env bash
set -euo pipefail

source "$HOME/capstone/.venv/bin/activate"
source "$HOME/capstone/env.sh"

exec python3 "$HOME/capstone/caption_feedback.py" \
  --model "$HOME/capstone/yolo11n_fp16.engine" \
  --camera 0 \
  --device 0 \
  --conf 0.25 \
  --iou 0.45 \
  --show \
  --speak \
  --speak-every 6
  # add this if/when you have Places365 files:
  # --places-dir "$HOME/capstone/places365"

