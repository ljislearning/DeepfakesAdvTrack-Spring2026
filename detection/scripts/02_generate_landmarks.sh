#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
FACE_ROOT=${FACE_ROOT:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face}

"${PYTHON}" data_preparation/extract_face/generate_landmarks_dlib_celeb_df_v2.py \
  --image_root_path "${FACE_ROOT}"
