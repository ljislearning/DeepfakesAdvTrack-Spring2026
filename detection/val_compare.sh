set -e

ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection
PYTHON=/home/duyijie/.conda/envs/course_AISA/bin/python
VAL_DATA=$ROOT/dataset/val
RESULT_DIR=$ROOT/results/val_compare
GT_PATH=$VAL_DATA/val_gts.xlsx

BASELINE_NAME=Hardworking111_baseline_val
UCF_NAME=Hardworking111_ucf_val
UCF_CKPT=/home/duyijie/DeepfakeBench/logs/training/ucf_2025-04-27-09-32-28/test/avg/ckpt_best.pth

mkdir -p "$RESULT_DIR"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference.py \
    --your-team-name "$BASELINE_NAME" \
    --data-folder "$VAL_DATA" \
    --model-weights "$ROOT/utils/weights.ckpt" \
    --result-path "$RESULT_DIR"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference_ucf.py \
    --your-team-name "$UCF_NAME" \
    --data-folder "$VAL_DATA" \
    --model-weights "$UCF_CKPT" \
    --result-path "$RESULT_DIR" \
    --deepfakebench-root /home/duyijie/DeepfakeBench

"$PYTHON" evaluate_val.py \
    --gt-path "$GT_PATH" \
    --submit-files "$RESULT_DIR/$BASELINE_NAME.xlsx" "$RESULT_DIR/$UCF_NAME.xlsx" \
    --output-path "$RESULT_DIR/val_compare.xlsx"
