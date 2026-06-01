# Hardworking111 TriGuard 技术报告

## 1. 方案概述

Hardworking111 TriGuard 是一个面向人脸深伪检测的三分类检测器。传统二分类只学习 real/fake 两个边界，本方案将 fake 进一步拆成两种训练目标：

```text
class 0: real face
class 1: clean fake face
class 2: artifact-augmented fake face
```

推理阶段将 class 1 和 class 2 合并为 fake，最终提交分数为：

```text
fake_score = 1 - P(real)
```

这样可以让模型同时关注伪造纹理、压缩痕迹、模糊退化、分辨率变化和轻微噪声等线索，而不是只依赖单一数据集中的固定伪造模式。

## 2. 数据采样

训练数据使用 Celeb-DF 视频抽帧后的人脸裁剪图。目录中提供了 `prepare_celeb_faces.py`，可以从原视频中均匀抽取帧、使用 dlib 检测人脸、按 1.3 倍人脸框扩展为方形裁剪，并生成：

```text
train.csv
val.csv
```

每条 csv 记录包括裁剪图路径、标签、原视频名和帧号。训练时对 fake 样本额外复制若干份并施加 artifact augmentation，构造第三类。类别采样使用 `WeightedRandomSampler`，保证 real、clean fake、artifact fake 在训练中更均衡。

## 3. 预处理

训练和推理均使用人脸区域作为输入。对于课程测试集，输入目录包含：

```text
img_list.txt
face_info.txt
imgs/
```

推理脚本读取 `face_info.txt` 的人脸框，按 1.3 倍扩展后裁剪，再 resize 到 `300 x 300`。图像归一化使用 ImageNet 均值和方差：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 4. 数据增强

Hardworking111 TriGuard 的第三类不是简单复制 fake，而是对 fake 样本施加随机退化增强，包括：

- Gaussian blur
- 下采样再上采样
- JPEG 压缩
- Gaussian noise
- 亮度和对比度扰动

另外，普通训练增强包括随机裁剪、轻量颜色扰动和 tensor-level noise。相比只做水平翻转，这种增强能更好覆盖测试集中可能出现的压缩和画质变化。

## 5. 模型结构

模型主干为 EfficientNet-B3，分类头输出 3 类 logits。训练目标为三分类交叉熵，推理目标为二分类 fake score。

训练损失：

```text
CrossEntropyLoss(label_smoothing=0.05)
```

优化器：

```text
SGD(lr=1e-3, momentum=0.9, weight_decay=1e-5)
StepLR(step_size=2, gamma=0.9)
```

## 6. 推理策略

推理时使用 3-view TTA：

```text
original
vertical flip
horizontal flip
```

三次前向分别 softmax 后取平均，再计算 `1 - P(real)`。该 TTA 保留了稳定性，同时控制了推理时间。

## 7. 分数校准

验证集上模型排序能力较好，但原始 fake score 的默认 0.5 阈值并不是最优判定点。因此使用验证集扫描得到的阈值做 logit-shift 校准：

```text
threshold = 0.06568622589111328
```

校准公式：

```text
score_calibrated = sigmoid((logit(score_raw) - logit(threshold)) / temperature)
```

该操作不改变样本排序，因此 AUC 不变；但可以让 `0.5` 成为更合理的提交判定阈值。验证集效果：

```text
AUC = 0.945084
AP = 0.996307
acc@0.5 = 0.960500
```

## 8. 运行方式

完整训练推理流程：

```bash
cd /home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new

# 1. 从原始视频抽取人脸
VIDEO_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/Celeb \
FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
sh scripts/01_extract_faces.sh

# 2. 生成人脸 landmark json，用于 blending
FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
sh scripts/02_generate_landmarks.sh

# 3. 生成 fake artifact/adversarial subsets
FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
sh scripts/03_generate_adversarial.sh

# 4. 训练三分类检测器
FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
sh scripts/04_train_detector.sh

# 5. 推理 test1 并校准分数
WEIGHTS=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/checkpoints/hardworking111_native/best_auc.pth \
sh scripts/05_infer_test1.sh
```

快速使用当前目录中已有的检测器权重推理 test1：

```bash
cd /home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new
sh run_infer_test1.sh
```

输出文件：

```text
results/Hardworking111.xlsx
results/Hardworking111_calibrated.xlsx
```

推荐提交：

```text
results/Hardworking111_calibrated.xlsx
```

完整复训时需要准备：

```text
Celeb-DF 原始视频
人脸裁剪输出目录
dlib landmark json
Celeb-synthesis_adv1 ... Celeb-synthesis_adv5
data_preparation/generate_adversarial/weights/ 下的攻击基座模型权重
```

## 9. 实现细节修改

本实现做了几处工程化修改：

- 将数据预处理、模型结构、权重、训练、推理和校准全部放入 `detection_new`，避免依赖旧实验目录。
- 使用三分类训练，但提交时自动合并为二分类 fake score。
- 通过 fake artifact/adversarial subsets 替代单纯 fake/real 二分类，提升对画质变化和扰动样本的鲁棒性。
- 推理阶段使用 3-view TTA，在稳定性和速度之间折中。
- 使用 logit-shift 校准，使输出文件更适配以 `0.5` 为阈值的评估脚本。
