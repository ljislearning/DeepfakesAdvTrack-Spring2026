set -e

ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection
PYTHON=/home/duyijie/.conda/envs/course_AISA/bin/python
VAL_DATA=$ROOT/dataset/val
RESULT_DIR=$ROOT/results/val_ucf_ckpts
GT_PATH=$VAL_DATA/val_gts.xlsx

UCF_RUN=/home/duyijie/DeepfakeBench/logs/training/ucf_2025-04-27-09-32-28/test

mkdir -p "$RESULT_DIR"

run_baseline() {
    local name=baseline_official_val
    local out=$RESULT_DIR/$name.xlsx
    if [ -f "$out" ]; then
        echo "Skip existing $out"
    else
        CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference.py \
            --your-team-name "$name" \
            --data-folder "$VAL_DATA" \
            --model-weights "$ROOT/utils/weights.ckpt" \
            --result-path "$RESULT_DIR"
    fi
}

run_ucf() {
    local tag=$1
    local ckpt=$2
    local name=ucf_${tag}_val
    local out=$RESULT_DIR/$name.xlsx
    if [ -f "$out" ]; then
        echo "Skip existing $out"
    else
        CUDA_VISIBLE_DEVICES=0 "$PYTHON" inference_ucf.py \
            --your-team-name "$name" \
            --data-folder "$VAL_DATA" \
            --model-weights "$ckpt" \
            --result-path "$RESULT_DIR" \
            --deepfakebench-root /home/duyijie/DeepfakeBench
    fi
}

run_baseline
run_ucf avg "$UCF_RUN/avg/ckpt_best.pth"
run_ucf celebdfv1 "$UCF_RUN/Celeb-DF-v1/ckpt_best.pth"
run_ucf celebdfv2 "$UCF_RUN/Celeb-DF-v2/ckpt_best.pth"
run_ucf dfdc "$UCF_RUN/DFDC/ckpt_best.pth"
run_ucf dfdcp "$UCF_RUN/DFDCP/ckpt_best.pth"
run_ucf uadfv "$UCF_RUN/UADFV/ckpt_best.pth"

"$PYTHON" evaluate_val.py \
    --gt-path "$GT_PATH" \
    --submit-files "$RESULT_DIR"/*.xlsx \
    --output-path "$RESULT_DIR/summary.xlsx"
