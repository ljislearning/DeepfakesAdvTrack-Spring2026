#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/duyijie/.conda/envs/DeepfakeBench/bin/python}
GPU=${GPU:-0}
FACE_ROOT=${FACE_ROOT:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face}
OUTPUT_DIR=${OUTPUT_DIR:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/checkpoints/hardworking111_native}

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" train_hardworking111_native.py \
  --face-root "${FACE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 85 \
  --batch-size 8 \
  --val-batch-size 64
