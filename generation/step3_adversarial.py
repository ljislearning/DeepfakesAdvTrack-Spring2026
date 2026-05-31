#!/usr/bin/env python3
"""
Step 3: 对抗参数搜索

用自己的输出来跑 Xception 检测器，找到最优对抗后处理参数。
测试维度：JPEG 压缩质量 + Gaussian Blur + 不做处理(baseline)
"""

import os, sys, json, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image
import torch, torchvision.transforms as T

PROJECT = "/home/liji/projects/DeepfakesAdvTrack-Spring2026"
OUTPUT_DIR = os.path.join(PROJECT, "workspace/output")
DETECTION_DIR = os.path.join(PROJECT, "detection")

sys.path.insert(0, DETECTION_DIR)
from utils import Xception as XceptionModel

# ─── 加载 Xception 模型 ───
print("🔍 加载 Xception 检测器...")
weights_path = os.path.join(DETECTION_DIR, "utils/weights.ckpt")
device = "cuda:0"

model = XceptionModel()
model.fc = torch.nn.Linear(2048, 1)
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.eval().to(device)
print("   ✅ Xception 就绪")

# ─── 人脸检测（用 InsightFace 做 bbox） ───
print("🔍 加载人脸检测器...")
os.environ['INSIGHTFACE_HOME'] = os.path.expanduser('~/.insightface')
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("   ✅ 人脸检测就绪")

# ─── 收集输出图像 ───
img_files = sorted([
    f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')
])
print(f"📋 共 {len(img_files)} 张输出图像")

# ─── 对每张图检测人脸并裁剪 ───
transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def detect_and_crop(img_path):
    """检测人脸、裁剪、返回 tensor"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        # 无人脸 → 用整图
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        return transform(pil).unsqueeze(0)

    # 取最大的人脸
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h, w = img.shape[:2]
    # 1.3x 扩展裁剪
    cx, cy = (x1+x2)//2, (y1+y2)//2
    size = int(max(x2-x1, y2-y1) * 1.3)
    x1 = max(cx - size//2, 0)
    y1 = max(cy - size//2, 0)
    size = min(w - x1, size)
    size = min(h - y1, size)
    cropped = img[y1:y1+size, x1:x1+size]
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    return transform(pil).unsqueeze(0)

@torch.no_grad()
def predict_batch(image_paths, batch_size=32):
    """批量预测 Xception 输出（sigmoid 概率）"""
    predictions = {}
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Xception 检测"):
        batch_paths = image_paths[i:i+batch_size]
        tensors = []
        valid_paths = []
        for p in batch_paths:
            t = detect_and_crop(p)
            if t is not None:
                tensors.append(t)
                valid_paths.append(p)
        if not tensors:
            continue
        batch = torch.cat(tensors).to(device)
        preds = model(batch).cpu().numpy().flatten()
        for p, pred in zip(valid_paths, preds):
            predictions[os.path.basename(p)] = float(pred)
    return predictions

# ─── 定义对抗后处理策略 ───
def apply_postprocess(img, method, param):
    """对图像应用对抗后处理"""
    if method == "none":
        return img
    elif method == "jpeg":
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, param])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif method == "blur":
        k = max(1, param)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)
    elif method == "jpeg+blur":
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, param[0]])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        k = max(1, param[1])
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)
    return img

# ─── 参数搜索 ───
strategies = [
    ("none", None),           # baseline
    ("jpeg", 95),
    ("jpeg", 85),
    ("jpeg", 75),
    ("jpeg", 65),
    ("blur", 3),
    ("blur", 5),
    ("blur", 7),
    ("jpeg+blur", (85, 3)),
    ("jpeg+blur", (75, 3)),
    ("jpeg+blur", (85, 5)),
]

TEMP_DIR = os.path.join(PROJECT, "workspace/temp_adv")
os.makedirs(TEMP_DIR, exist_ok=True)

results = []

print("\n🔬 开始参数搜索...\n")
for method, param in strategies:
    label = f"{method}_{param}" if param else method
    
    # 生成后处理版本
    temp_img_dir = os.path.join(TEMP_DIR, label)
    os.makedirs(temp_img_dir, exist_ok=True)
    
    paths = []
    for f in img_files[:200]:  # 采样 200 张加速搜索
        src = os.path.join(OUTPUT_DIR, f)
        dst = os.path.join(temp_img_dir, f)
        if not os.path.exists(dst):
            img = cv2.imread(src)
            processed = apply_postprocess(img, method, param)
            cv2.imwrite(dst, processed)
        paths.append(dst)
    
    # 检测
    preds = predict_batch(paths)
    
    # 计算平均预测分数（越低越好，越低说明越像真图）
    scores = list(preds.values())
    avg_score = np.mean(scores)
    
    # 假设 ground truth 全是 fake=1（我们生成的都是假图）
    # 如果检测器判为 0（真），说明对抗成功
    fool_rate = np.mean([1 for s in scores if s < 0.5])
    
    print(f"  {label:20s} | avg_score={avg_score:.4f} | fool_rate={fool_rate:.2%}")
    results.append({
        "strategy": label,
        "avg_score": avg_score,
        "fool_rate": fool_rate,
    })

# ─── 排序输出 ───
print("\n" + "="*60)
print("📊 对抗策略排名（avg_score 越低越能骗过检测器）")
print("="*60)
results.sort(key=lambda x: x["avg_score"])
for i, r in enumerate(results):
    bar = "█" * int(r["fool_rate"] * 30)
    print(f"  {i+1}. {r['strategy']:20s}  avg={r['avg_score']:.4f}  fool={r['fool_rate']:.1%}  {bar}")

best = results[0]
print(f"\n🏆 最佳策略: {best['strategy']} (avg_score={best['avg_score']:.4f})")
print("\n💡 确认最佳策略后，用它对全部 1000 张做后处理即可提交。")
