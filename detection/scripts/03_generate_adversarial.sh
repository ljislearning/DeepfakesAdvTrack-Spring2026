#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
GPU=${GPU:-0}
FACE_ROOT=${FACE_ROOT:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face}
FAKE_ROOT="${FACE_ROOT}/Celeb-synthesis"

if [ ! -d "data_preparation/generate_adversarial/weights" ]; then
  echo "Missing attack weights: data_preparation/generate_adversarial/weights" >&2
  echo "Download or train the attack baseline weights before running adversarial generation." >&2
  exit 1
fi

cd data_preparation/generate_adversarial

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" attack_ensemble_example1.py --gpu_id "${GPU}" --input_path "${FAKE_ROOT}"
CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" attack_ensemble_example2.py --gpu_id "${GPU}" --input_path "${FAKE_ROOT}" --use_mask
CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" attack_ensemble_example3.py --gpu_id "${GPU}" --input_path "${FAKE_ROOT}" --use_mask
CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" attack_ensemble_example4.py --gpu_id "${GPU}" --input_path "${FAKE_ROOT}" --use_mask
CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" attack_ensemble_example5.py --gpu_id "${GPU}" --input_path "${FAKE_ROOT}" --use_mask
