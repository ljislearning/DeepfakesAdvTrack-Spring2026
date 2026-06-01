#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/duyijie/.conda/envs/course_AISA/bin/python}
GPU=${GPU:-2}
TEAM=${TEAM:-Hardworking111}
DATA_FOLDER=${DATA_FOLDER:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/test1}
WEIGHTS=${WEIGHTS:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/weights/hardworking111_triguard_b3.pth}
RESULT_DIR=${RESULT_DIR:-/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/results}
THRESHOLD=${THRESHOLD:-0.06568622589111328}

export PYTHON GPU TEAM DATA_FOLDER WEIGHTS RESULT_DIR THRESHOLD
sh scripts/05_infer_test1.sh
