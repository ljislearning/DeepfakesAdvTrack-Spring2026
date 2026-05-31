set -e

ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection
PYTHON=/home/duyijie/.conda/envs/course_AISA/bin/python

cd "$ROOT"

CUDA_VISIBLE_DEVICES=2,3,4,5 "$PYTHON" train_baseline_ucf_celeb.py \
    --data-root "$ROOT/dataset/Celeb" \
    --init-weights "$ROOT/utils/weights.ckpt" \
    --output-dir "$ROOT/checkpoints/baseline_ucf_celeb_fair_xception" \
    --epochs 4 \
    --batch-size 32 \
    --frames-per-video 6 \
    --lr 3e-6 \
    --num-workers 12 \
    --multi-gpu \
    --no-ucf-augment \
    --freeze-backbone \
    --unfreeze-last-block \
    --balanced-sampler \
    --label-smoothing 0.05 \
    --strong-augment
