#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/duyijie/.conda/envs/course_AISA/bin/python}
GPU=${GPU:-2}
TEAM=${TEAM:-Hardworking111}
DATA_FOLDER=${DATA_FOLDER:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/test1}
WEIGHTS=${WEIGHTS:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/weights/hardworking111_triguard_b3.pth}
RESULT_DIR=${RESULT_DIR:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/results}
THRESHOLD=${THRESHOLD:-0.06568622589111328}

CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" infer_hardworking111.py \
  --your-team-name "${TEAM}" \
  --data-folder "${DATA_FOLDER}" \
  --model-weights "${WEIGHTS}" \
  --result-path "${RESULT_DIR}" \
  --batch-size 20 \
  --num-workers 4

"${PYTHON}" calibrate_hardworking111.py \
  --input "${RESULT_DIR}/${TEAM}.xlsx" \
  --output "${RESULT_DIR}/${TEAM}_calibrated.xlsx" \
  --threshold "${THRESHOLD}"
