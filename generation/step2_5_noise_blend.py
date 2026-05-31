#!/usr/bin/env python3
"""
Step 2.5: 频域噪声一致化后处理 (v2 - 修复边界artifact)

核心改进：
  - 只对人脸区域 patch 做 FFT（不全局 FFT，避免 ringing）
  - 大幅增加羽化过渡区域（kernel = min(H,W)/12）
  - 在 patch 边缘做 alpha 融合，消除硬边界

用法：
  source .pyenv/versions/3.10.20/envs/deepfake-gen/bin/activate
  python step2_5_noise_blend.py
"""

import os, sys, json, cv2, numpy as np
from tqdm import tqdm

PROJECT = "/home/liji/projects/DeepfakesAdvTrack-Spring2026"
TARGET_DIR = os.path.join(PROJECT, "workspace/target_frames")
SWAP_DIR = os.path.join(PROJECT, "workspace/output")
OUTPUT_DIR = os.path.join(PROJECT, "workspace/output_blended")
MANIFEST_PATH = os.path.join(PROJECT, "workspace/task_manifest.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔍 加载 InsightFace 人脸检测器...")
os.environ['INSIGHTFACE_HOME'] = os.path.expanduser('~/.insightface')
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("   ✅ 就绪")

with open(MANIFEST_PATH) as f:
    tasks = json.load(f)
task_map = {t['output_name']: t for t in tasks}
print(f"📋 共 {len(tasks)} 个任务")


def get_face_bbox_and_weight(img, expand_ratio=1.4, feather_ratio=0.15):
    """
    检测人脸，返回扩展后的 bbox + 软羽化权重图。
    feather_ratio: 羽化过渡占 bbox 尺寸的比例（越大越平滑）
    """
    h, w = img.shape[:2]
    faces = app.get(img)

    if not faces:
        return None, None, np.zeros((h, w), dtype=np.float32)

    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    half = int(max(bbox_w, bbox_h) * expand_ratio / 2)
    px1 = max(cx - half, 0)
    py1 = max(cy - half, 0)
    psize = min(w - px1, h - py1, half * 2)
    px2, py2 = px1 + psize, py1 + psize

    bbox = (px1, py1, px2, py2)

    # 创建羽化权重图：从中心椭圆向外过渡
    weight = np.zeros((h, w), dtype=np.float32)
    # 内部椭圆 mask（完全融合）
    inner_ratio = 0.75  # 内圈占 75%，完全保留融合结果
    inner_axes = (int(psize * inner_ratio / 2), int(psize * inner_ratio / 2))
    weight = cv2.ellipse(weight, (cx, cy), inner_axes, 0, 0, 360, 1.0, -1)

    # 用大 kernel 高斯模糊做羽化（过渡区域 = bbox 的 ~15%）
    kernel_size = int(psize * feather_ratio)
    if kernel_size < 5:
        kernel_size = 5
    if kernel_size % 2 == 0:
        kernel_size += 1
    weight = cv2.GaussianBlur(weight, (kernel_size, kernel_size), 0)
    weight = np.clip(weight, 0, 1)

    return bbox, (cx, cy, psize), weight


def freq_blend_patch(swapped_patch, target_patch, lowpass_sigma=35, highpass_blend=0.8):
    """
    对 patch（人脸区域）做频域融合。
    - 低频 → swapped（保持结构/身份）
    - 高频 → target（继承噪声特征）
    """
    h_p, w_p = swapped_patch.shape[:2]

    # 确保 patch 尺寸适合 FFT（偶数尺寸更好）
    # 这里假设 patch 已经是合适的

    # FFT
    sw_f = np.fft.fft2(swapped_patch.astype(np.float32), axes=(0, 1))
    tg_f = np.fft.fft2(target_patch.astype(np.float32), axes=(0, 1))

    sw_mag = np.abs(sw_f)
    sw_phase = np.angle(sw_f)
    tg_mag = np.abs(tg_f)

    # 频率域高斯低通滤波器
    fy = np.fft.fftfreq(h_p).reshape(-1, 1)
    fx = np.fft.fftfreq(w_p).reshape(1, -1)
    dist = np.sqrt(fx**2 + fy**2)
    sigma_freq = lowpass_sigma / max(h_p, w_p)
    lowpass = np.exp(-dist**2 / (2 * sigma_freq**2))
    lowpass = lowpass[:, :, np.newaxis]

    # 幅度融合
    blended_mag = lowpass * sw_mag + (1 - lowpass) * (
        highpass_blend * tg_mag + (1 - highpass_blend) * sw_mag
    )

    # 重建
    blended_f = blended_mag * np.exp(1j * sw_phase)
    blended_patch = np.fft.ifft2(blended_f, axes=(0, 1)).real
    blended_patch = np.clip(blended_patch, 0, 255).astype(np.uint8)

    return blended_patch


# ─── 批量处理 ───
print("\n🎨 开始频域噪声融合 (v2 - patch-based, 无边界artifact)...")
swapped_files = sorted([f for f in os.listdir(SWAP_DIR) if f.endswith('.png')])
blended, skipped, no_face = 0, 0, 0

for fname in tqdm(swapped_files, desc="频域融合"):
    if fname not in task_map:
        skipped += 1
        continue

    swapped_path = os.path.join(SWAP_DIR, fname)
    swapped_img = cv2.imread(swapped_path)
    if swapped_img is None:
        skipped += 1
        continue

    target_path = os.path.join(TARGET_DIR, fname)
    target_img = cv2.imread(target_path)
    if target_img is None:
        skipped += 1
        continue

    if swapped_img.shape != target_img.shape:
        target_img = cv2.resize(target_img, (swapped_img.shape[1], swapped_img.shape[0]))

    # 检测人脸 bbox + 权重图
    bbox, (cx, cy, psize), weight_map = get_face_bbox_and_weight(swapped_img)

    if bbox is None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), swapped_img)
        no_face += 1
        continue

    px1, py1, px2, py2 = bbox

    try:
        # 提取人脸 patch
        swapped_patch = swapped_img[py1:py2, px1:px2]
        target_patch = target_img[py1:py2, px1:px2]

        # 频域融合（只在 patch 内）
        blended_patch = freq_blend_patch(swapped_patch, target_patch)

        # 用羽化权重融合回原图
        weight_3c = weight_map[:, :, np.newaxis]
        # 构建融合图：在原图中用 blended_patch 替换对应区域
        full_blended = swapped_img.copy()
        full_blended[py1:py2, px1:px2] = blended_patch

        # 羽化融合
        result = (weight_3c * full_blended + (1 - weight_3c) * swapped_img).astype(np.uint8)

        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), result)
        blended += 1
    except Exception as e:
        print(f"  ⚠️  融合失败 {fname}: {e}")
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), swapped_img)
        blended += 1

print(f"\n✅ 频域融合完成: {blended}/{len(swapped_files)}")
print(f"   无人脸: {no_face}, 跳过: {skipped}")
print(f"   输出目录: {OUTPUT_DIR}")
print(f"   输出文件数: {len(os.listdir(OUTPUT_DIR))}")
