#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/duyijie/.conda/envs/DeepfakeBench/bin/python}
GPU=${GPU:-0}
FACE_ROOT=${FACE_ROOT:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face}

export PYTHON GPU FACE_ROOT
sh scripts/04_train_detector.sh
