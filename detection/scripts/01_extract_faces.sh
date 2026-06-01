#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
GPU=${GPU:-0}
VIDEO_ROOT=${VIDEO_ROOT:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/Celeb}
FACE_ROOT=${FACE_ROOT:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face}

mkdir -p "${FACE_ROOT}"

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" data_preparation/extract_face/extract_video_celeb_df_v2.py \
  --gpu_id "${GPU}" \
  --video_root_path "${VIDEO_ROOT}" \
  --image_root_path "${FACE_ROOT}"

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" data_preparation/extract_face/extract_video_celeb_df_v2_yotube.py \
  --gpu_id "${GPU}" \
  --video_root_path "${VIDEO_ROOT}" \
  --image_root_path "${FACE_ROOT}"
