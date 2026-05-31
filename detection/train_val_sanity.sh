set -e

ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection
PYTHON=/home/duyijie/.conda/envs/course_AISA/bin/python

cd "$ROOT"

CUDA_VISIBLE_DEVICES=2,3,4,5 "$PYTHON" train_val_sanity.py \
    --data-folder "$ROOT/dataset/val" \
    --gt-path "$ROOT/dataset/val/val_gts.xlsx" \
    --init-weights "$ROOT/utils/weights.ckpt" \
    --output-dir "$ROOT/checkpoints/val_sanity" \
    --epochs 3 \
    --batch-size 64 \
    --lr 1e-5 \
    --num-workers 12 \
    --multi-gpu
