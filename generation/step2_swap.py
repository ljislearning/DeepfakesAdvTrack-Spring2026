#!/usr/bin/env python3
"""
Step 2: 批量换脸

为每个 source identity 选出最佳人脸（FaceAnalysis 检测到的最大置信度人脸），
然后对每个 target 帧检测人脸并换脸。
使用 inswapper_128 + paste_back 自动融合。
"""

import os, sys, json, cv2, numpy as np
from tqdm import tqdm

PROJECT = "/home/liji/projects/DeepfakesAdvTrack-Spring2026"
TARGET_DIR = os.path.join(PROJECT, "workspace/target_frames")
SOURCE_DIR = os.path.join(PROJECT, "workspace/source_faces")
OUTPUT_DIR = os.path.join(PROJECT, "workspace/output")
MANIFEST_PATH = os.path.join(PROJECT, "workspace/task_manifest.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ['INSIGHTFACE_HOME'] = os.path.expanduser('~/.insightface')

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ─── 初始化模型 (GPU 5) ───
print("🚀 加载 InsightFace 模型...")
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = get_model(
    os.path.expanduser('~/.insightface/models/inswapper_128.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
print(f"   ✅ inswapper 就绪: {swapper.input_size}")

# ─── 任务清单 ───
with open(MANIFEST_PATH) as f:
    tasks = json.load(f)
print(f"📋 共 {len(tasks)} 个换脸任务")

# ─── 为每个 source identity 选最佳人脸 ───
print("\n🔍 为每个 identity 选择最佳 source 人脸...")
source_faces = {}  # {identity: (best_face_img, face_obj)}

for sid in sorted(set(t['source_id'] for t in tasks)):
    sd = os.path.join(SOURCE_DIR, sid)
    candidates = [f for f in os.listdir(sd) if f.endswith(('.png', '.jpg'))]
    if not candidates:
        print(f"   ⚠️  {sid}: 无候选图")
        continue

    best_face = None
    best_score = -1
    for cf in candidates:
        img = cv2.imread(os.path.join(sd, cf))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        # 选检测置信度最高的人脸
        top = max(faces, key=lambda f: f.det_score)
        if top.det_score > best_score:
            best_score = top.det_score
            best_face = top

    if best_face is not None:
        source_faces[sid] = best_face
        print(f"   ✅ {sid}: 最佳人脸 score={best_score:.3f}")
    else:
        print(f"   ⚠️  {sid}: 未检测到人脸")

print(f"\n   {len(source_faces)}/{len(set(t['source_id'] for t in tasks))} 个 identity 有 source 人脸")

# ─── 批量换脸 ───
print("\n🎭 开始批量换脸...")
swapped = 0
failed = 0

for t in tqdm(tasks, desc="换脸进度"):
    output_name = t['output_name']
    source_id = t['source_id']

    if source_id not in source_faces:
        failed += 1
        continue

    # 读取 target 帧
    target_path = os.path.join(TARGET_DIR, output_name)
    target_img = cv2.imread(target_path)
    if target_img is None:
        failed += 1
        continue

    # 检测 target 人脸
    target_faces = app.get(target_img)
    if not target_faces:
        # 没有人脸 → 直接复制原图
        cv2.imwrite(os.path.join(OUTPUT_DIR, output_name), target_img)
        swapped += 1
        continue

    # 对每张检测到的人脸换为 source 的脸
    src_face = source_faces[source_id]
    result = target_img.copy()

    for face in target_faces:
        try:
            result = swapper.get(result, face, src_face, paste_back=True)
        except Exception as e:
            print(f"   ⚠️  换脸失败 {output_name}: {e}")
            result = target_img.copy()
            break

    # 保存
    out_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(out_path, result)
    swapped += 1

print(f"\n✅ 换脸完成: {swapped}/{len(tasks)}")
print(f"   输出目录: {OUTPUT_DIR}")

# ─── 统计 ───
print(f"\n📊 输出文件数: {len(os.listdir(OUTPUT_DIR))}")
