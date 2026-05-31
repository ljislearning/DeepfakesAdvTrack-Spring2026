#!/usr/bin/env python3
"""
Step 3 (v4 - delta上采样): PGD对抗攻击，扰动保存然后上采样叠加

v4 核心改动（方案B）:
  旧: PGD在299×299生成对抗图 → upscale全图 → 融合 → 检测器resize回来时扰动被抹掉
  新: PGD在299×299生成delta → upscale delta到原图分辨率 → 叠加到原图人脸区域
      检测器resize(原图 + upscale(delta)) ≈ resize(原图) + delta ✅

用法：python step3_adv_pgd.py
"""

import os, sys, cv2, numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torch.nn as nn
import torchvision.transforms as T

PROJECT = "/home/liji/projects/DeepfakesAdvTrack-Spring2026"
INPUT_DIR = os.path.join(PROJECT, "workspace/output_blended")
if not os.path.isdir(INPUT_DIR):
    INPUT_DIR = os.path.join(PROJECT, "workspace/output")

DETECTION_DIR = os.path.join(PROJECT, "detection")
OUTPUT_BASE = os.path.join(PROJECT, "workspace/output_adv")

sys.path.insert(0, DETECTION_DIR)
from utils import Xception as XceptionModel

device = "cuda:0"

# ─── 加载检测器 ───
print("🔍 加载 Xception 检测器...")
weights_path = os.path.join(DETECTION_DIR, "utils/weights.ckpt")
model = XceptionModel()
model.fc = nn.Linear(2048, 1)
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.eval().to(device)
for p in model.parameters():
    p.requires_grad = False
print("   ✅ 就绪")

# ─── 人脸检测 ───
print("🔍 加载人脸检测器...")
os.environ['INSIGHTFACE_HOME'] = os.path.expanduser('~/.insightface')
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("   ✅ 就绪")

transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def make_smooth_face_weight(img, expand_ratio=1.35, feather_ratio=0.15):
    """生成极平滑的人脸羽化权重"""
    h, w = img.shape[:2]
    faces = app.get(img)
    if not faces:
        return np.zeros((h, w), dtype=np.float32), None

    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bbox_w, bbox_h = x2 - x1, y2 - y1
    half = int(max(bbox_w, bbox_h) * expand_ratio / 2)
    half = min(half, cx, w - cx, cy, h - cy)
    half = max(half, 4)
    psize = half * 2

    inner_size = max(2, int(psize * 0.7))
    weight = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(weight, (cx, cy), (inner_size//2, inner_size//2), 0, 0, 360, 1.0, -1)

    feather_kernel = int(psize * feather_ratio)
    feather_kernel = max(21, feather_kernel)
    if feather_kernel % 2 == 0:
        feather_kernel += 1
    weight = cv2.GaussianBlur(weight, (feather_kernel, feather_kernel), 0)
    weight = np.clip(weight, 0, 1)

    # 人脸 patch 的 bbox
    px1 = max(cx - half, 0)
    py1 = max(cy - half, 0)
    px2 = min(px1 + psize, w)
    py2 = min(py1 + psize, h)
    bbox = (px1, py1, px2, py2)

    return weight, bbox


def pgd_attack_delta(img_bgr, face_weight, face_bbox,
                     eps=6.0/255, alpha=1.0/255, steps=40):
    """
    方案B PGD: 在299×299上生成delta，上采样delta到原图分辨率叠加。

    Returns: (result_bgr, before_score, after_score)
    """
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0

    px1, py1, px2, py2 = face_bbox
    face_h = py2 - py1
    face_w = px2 - px1

    if face_weight.shape[0] != h_orig or face_weight.shape[1] != w_orig:
        face_weight = cv2.resize(face_weight, (w_orig, h_orig))
    face_weight_3c = face_weight[:, :, np.newaxis]

    # ─── 1. 提取人脸 patch，缩放到 299×299，做 PGD ───
    face_patch = img_float[py1:py2, px1:px2]  # (face_h, face_w, 3)
    face_299 = cv2.resize((face_patch * 255).astype(np.uint8), (299, 299)).astype(np.float32) / 255.0

    x = torch.from_numpy(face_299.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    x = (x - 0.5) / 0.5  # → [-1, 1]
    x_orig = x.clone()

    with torch.no_grad():
        before_score = model(x_orig).item()

    delta = torch.zeros_like(x_orig, requires_grad=True)
    for _ in range(steps):
        x_adv = x_orig + delta
        loss = model(x_adv).mean()
        loss.backward()
        with torch.no_grad():
            delta = delta - alpha * delta.grad.sign()
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.clamp(x_orig + delta, -1, 1) - x_orig
        delta.requires_grad = True

    x_final = x_orig + delta.detach()
    with torch.no_grad():
        after_score = model(x_final).item()

    # ─── 2. 提取 delta（[-1,1] 空间），转到 [0,255] 像素空间 ───
    delta_pixel = denorm(x_orig + delta.detach()) - denorm(x_orig)
    delta_pixel = delta_pixel.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (299, 299, 3)
    delta_pixel = (delta_pixel * 255).astype(np.float32)

    # ─── 3. 上采样 delta 到人脸 patch 尺寸，叠加到原图 ───
    delta_face = cv2.resize(delta_pixel, (face_w, face_h), interpolation=cv2.INTER_LINEAR)
    delta_face = delta_face / 255.0

    # 叠加到原图人脸区域
    result = img_float.copy()
    result[py1:py2, px1:px2] = np.clip(
        result[py1:py2, px1:px2] + delta_face, 0, 1
    )

    # ─── 4. 羽化融合 + 双边滤波 ───
    # 人脸区域用权重做软过渡
    result_full = face_weight_3c * result + (1 - face_weight_3c) * img_float
    result_full = (result_full * 255).astype(np.uint8)

    # 双边滤波（保边去噪）
    result_smooth = cv2.bilateralFilter(result_full, d=9, sigmaColor=50, sigmaSpace=50)
    result_smooth = result_smooth.astype(np.float32)

    result_final = face_weight_3c * result_smooth + (1 - face_weight_3c) * result_full.astype(np.float32)
    result_final = np.clip(result_final, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_final, cv2.COLOR_RGB2BGR)

    return result_bgr, (before_score, after_score)


# ─── 参数搜索 ───
img_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])
print(f"📋 共 {len(img_files)} 张输入图像")

print("\n" + "="*60)
print("🔬 参数搜索 (sample=150, delta上采样方案)...")
print("="*60)

# 用稍大 eps 补偿 resize 衰减
search_configs = [
    (4/255, 0.5/255, 20, "eps=4 α=0.5 steps=20"),
    (4/255, 1.0/255, 20, "eps=4 α=1.0 steps=20"),
    (6/255, 0.5/255, 30, "eps=6 α=0.5 steps=30"),
    (6/255, 1.0/255, 30, "eps=6 α=1.0 steps=30"),
    (8/255, 0.5/255, 40, "eps=8 α=0.5 steps=40"),
    (8/255, 1.0/255, 40, "eps=8 α=1.0 steps=40"),
]

sample_n = min(150, len(img_files))
sample_files = img_files[:sample_n]
best_config, best_avg_score = None, 999

for eps, alpha, steps, label in search_configs:
    scores_before, scores_after = [], []
    for fname in tqdm(sample_files, desc=f"  {label}", leave=False):
        img_bgr = cv2.imread(os.path.join(INPUT_DIR, fname))
        if img_bgr is None:
            continue
        face_weight, face_bbox = make_smooth_face_weight(img_bgr)
        if face_bbox is None:
            continue
        try:
            _, (b, a) = pgd_attack_delta(img_bgr, face_weight, face_bbox,
                                         eps=eps, alpha=alpha, steps=steps)
            scores_before.append(b)
            scores_after.append(a)
        except:
            continue

    avg_before = np.mean(scores_before) if scores_before else 0
    avg_after = np.mean(scores_after) if scores_after else 0
    fool_rate = np.mean([1 for s in scores_after if s < 0.5]) if scores_after else 0
    print(f"  {label:30s} | before={avg_before:.4f} → after={avg_after:.4f} | fool={fool_rate:.1%}")
    if avg_after < best_avg_score:
        best_avg_score = avg_after
        best_config = (eps, alpha, steps, label)

print(f"\n🏆 最佳: {best_config[3]} (after={best_avg_score:.4f})")

# ─── 全量处理 ───
best_eps, best_alpha, best_steps, best_label = best_config
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "delta_" + best_label.replace(" ", "_").replace("=", ""))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n🎯 全量处理 {len(img_files)} 张...")
all_before, all_after, processed = [], [], 0

for fname in tqdm(img_files, desc="PGD delta"):
    img_bgr = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img_bgr is None:
        continue
    face_weight, face_bbox = make_smooth_face_weight(img_bgr)
    if face_bbox is None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img_bgr)
        processed += 1
        continue
    try:
        adv_bgr, (b, a) = pgd_attack_delta(img_bgr, face_weight, face_bbox,
                                           eps=best_eps, alpha=best_alpha, steps=best_steps)
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), adv_bgr)
        all_before.append(b)
        all_after.append(a)
        processed += 1
    except Exception as e:
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img_bgr)
        processed += 1

print(f"\n✅ 完成: {processed}/{len(img_files)}")
print(f"   输出: {OUTPUT_DIR} ({len(os.listdir(OUTPUT_DIR))} 文件)")

if all_before:
    print(f"\n📊 效果 (Xception on 299×299 face patch):")
    print(f"   攻击前: {np.mean(all_before):.4f}")
    print(f"   攻击后: {np.mean(all_after):.4f}")
    print(f"   逃逸率: {np.mean([1 for s in all_after if s < 0.5]):.1%}")
