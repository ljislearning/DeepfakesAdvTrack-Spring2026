#!/usr/bin/env python3
"""
Step 1: 读取 image_list.txt，从 Celeb-real 视频中提取目标帧，
        同时为每个 identity 收集 source 人脸素材。

image_list 格式: id0_id1_0000_00060.png
  = target视频(id0_0000.mp4)的第60帧, 换source(id1)的脸
"""

import os, sys, cv2, json
from collections import defaultdict
from tqdm import tqdm

PROJECT = "/home/liji/projects/DeepfakesAdvTrack-Spring2026"
CELEB_REAL = os.path.join(PROJECT, "dataset/Celeb-DF-v2/Celeb-real")
IMAGE_LIST = os.path.join(PROJECT, "generation/image_list.txt")
TARGET_DIR = os.path.join(PROJECT, "workspace/target_frames")
SOURCE_DIR = os.path.join(PROJECT, "workspace/source_faces")

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)

# ─── 解析 image_list.txt ───
# 格式: {target_id}_{source_id}_{vid}_{frame}.png
# target_video = {target_id}_{vid}.mp4  (在 Celeb-real 中)
# source_id = identity 编号

tasks = []
with open(IMAGE_LIST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name = line.replace(".png", "")
        parts = name.split("_")
        # id0_id1_0000_00060 → target_id=id0, source_id=id1, vid=0000, frame=00060
        target_id = parts[0]
        source_id = parts[1]
        vid = parts[2]
        frame_num = int(parts[3])
        tasks.append({
            "output_name": line.strip(),
            "target_id": target_id,
            "source_id": source_id,
            "video_file": f"{target_id}_{vid}.mp4",
            "frame_num": frame_num,
        })

print(f"共 {len(tasks)} 个任务")
print(f"涉及 target_ids: {len(set(t['target_id'] for t in tasks))} 个")
print(f"涉及 source_ids: {len(set(t['source_id'] for t in tasks))} 个")

# ─── 提取目标帧 ───
# 按视频分组，避免重复打开同一个视频
video_tasks = defaultdict(list)
for t in tasks:
    video_tasks[t["video_file"]].append(t)

extracted = 0
missed = 0

for video_file, task_list in tqdm(video_tasks.items(), desc="提取目标帧"):
    video_path = os.path.join(CELEB_REAL, video_file)
    if not os.path.exists(video_path):
        print(f"  ⚠️  缺失: {video_file}")
        missed += len(task_list)
        continue

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    for t in task_list:
        frame_num = t["frame_num"]
        if frame_num >= total_frames:
            # 用最后一帧
            frame_num = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(TARGET_DIR, t["output_name"])
            cv2.imwrite(out_path, frame)
            t["extracted"] = True
            extracted += 1
        else:
            t["extracted"] = False
            missed += 1

    cap.release()

print(f"\n✅ 提取 {extracted} 张 | ❌ 缺失/失败 {missed} 张")

# ─── 从 Celeb-real 为每个 source identity 收集人脸 ───
# 策略：取每个 identity 第一个视频的第一帧和中间帧，存为 source 候选
# 正式换脸时会用人脸检测选最佳的一张

print("\n收集 source 人脸素材...")
source_identities = set(t["source_id"] for t in tasks)
collected = 0

for sid in tqdm(source_identities, desc="source 人脸"):
    save_dir = os.path.join(SOURCE_DIR, sid)
    os.makedirs(save_dir, exist_ok=True)
    if os.listdir(save_dir):
        continue  # 已收集过

    # 找该 identity 的所有视频
    videos = [f for f in os.listdir(CELEB_REAL)
              if f.startswith(f"{sid}_") and f.endswith(".mp4")]
    if not videos:
        print(f"  ⚠️  {sid} 无视频")
        continue

    # 取第一个视频的几个代表性帧
    video_path = os.path.join(CELEB_REAL, videos[0])
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 取 0%, 25%, 50%, 75% 位置的帧（不同角度/表情）
    for pct in [0, 0.25, 0.5, 0.75]:
        fn = max(0, min(int(total * pct), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(save_dir, f"frame_{int(total*pct):06d}.png"), frame)
            collected += 1

    cap.release()

print(f"收集了 {collected} 张 source 候选图（{len(source_identities)} 个身份）")

# ─── 保存任务清单 ───
manifest_path = os.path.join(PROJECT, "workspace/task_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)
print(f"\n任务清单已保存: {manifest_path}")
print("Step 1 完成 ✅")
