CUDA_VISIBLE_DEVICES=0 python inference_ucf.py \
    --your-team-name Hardworking111 \
    --data-folder /home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/test1 \
    --model-weights /home/duyijie/DeepfakeBench/logs/training/ucf_2025-04-27-09-32-28/test/avg/ckpt_best.pth \
    --result-path /home/duyijie/DeepfakesAdvTrack-Spring2026/detection/results \
    --deepfakebench-root /home/duyijie/DeepfakeBench
