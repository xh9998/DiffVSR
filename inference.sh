#!/usr/bin/env bash
set -euo pipefail

# Run the optimized inference script; override CUDA_VISIBLE_DEVICES externally as needed.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python inference_tile.py \
  -i /mnt/nas/vsr_test/Real1/Aurora \
  -o ./output/ \
  -txt /mnt/nas/vsr_test/CSV/RealVideo10_caption.csv \
  --use_ffmpeg
