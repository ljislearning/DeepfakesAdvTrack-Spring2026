#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

sh scripts/01_extract_faces.sh
sh scripts/02_generate_landmarks.sh
sh scripts/03_generate_adversarial.sh
sh scripts/04_train_detector.sh
sh scripts/05_infer_test1.sh
