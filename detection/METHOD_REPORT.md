# Deepfake Detection 方法报告

## 1. 方法概述

本方案以官方 starterkit 中的 Xception baseline 为基础，在不使用比赛验证集标签进行训练的前提下，仅使用 `/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/Celeb` 作为训练数据进行微调。最终提交文件为：

`/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/results/Hardworking111_dfgcaug.xlsx`

最终模型权重为：

`/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/checkpoints/baseline_ucf_celeb_fair_xception_dfgcaug/best_auc.ckpt`

核心改进点是：在 Celeb 数据上进行保守微调，同时加入 DFGC 风格的图像退化增强，包括 JPEG 压缩、模糊、降采样和轻微噪声，从而提升模型对比赛数据中后处理、压缩和扰动的泛化能力。

## 2. 数据使用

训练数据只使用 Celeb 数据集：

`/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/Celeb`

其中标签定义如下：

| 子目录 | 标签 | 含义 |
|---|---:|---|
| `YouTube-real` | 0 | real |
| `Celeb-real` | 0 | real |
| `Celeb-synthesis` | 1 | fake |

代码中没有使用课程验证集 `dataset/val/val_gts.xlsx` 参与训练。该验证集只用于训练后评估不同模型方案的泛化效果。

本次训练脚本自动按照类别分层划分 Celeb 数据：

| 划分 | 视频数 | real | fake |
|---|---:|---:|---:|
| 总计 | 1203 | 408 | 795 |
| 训练 | 1023 | 347 | 676 |
| Celeb 内部验证 | 180 | 61 | 119 |

每个视频采样 `6` 个 frame slot，因此训练样本展开后约为：

| 划分 | real frame items | fake frame items |
|---|---:|---:|
| 训练 | 2082 | 4056 |

由于 fake 样本更多，训练时使用 `WeightedRandomSampler` 进行类别均衡采样，使 real/fake 在训练 batch 中更均衡。

## 3. 数据采样策略

训练集使用视频级样本，每个视频构造 `frames_per_video=6` 个训练 item。每次读取训练 item 时，并不固定读取某一帧，而是在视频总帧数范围内随机抽取一帧。这样同一个视频在不同 epoch 中可能读到不同帧，可以增加训练多样性。

内部验证集使用确定性采样。对于每个视频，将 `6` 个 frame slot 均匀映射到视频时间轴上，读取固定位置的帧，保证验证结果可复现。

随机种子为 `2026`，用于固定数据划分和随机过程的初始状态。

## 4. 预处理与裁脸

训练阶段的输入来自视频帧。每一帧先经过 OpenCV 读取，然后进行人脸区域裁剪。

裁脸流程如下：

1. 使用 OpenCV Haar cascade `haarcascade_frontalface_default.xml` 检测人脸。
2. 如果检测到多张脸，选择面积最大的人脸框。
3. 以人脸框中心为中心，取正方形区域。
4. 裁剪框按 `face_scale=1.3` 放大，以保留部分脸部周围上下文。
5. 如果没有检测到人脸，则回退到图像中心正方形裁剪。
6. 将裁剪结果转换为 RGB PIL 图像。

训练和推理均使用 `299 x 299` 输入尺寸，并使用 Xception baseline 相同的归一化：

```python
Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

推理阶段使用 starterkit 提供的 `FolderDataset`。该数据集读取测试集中的 `face_info.txt`，按官方提供的人脸框进行 `1.3` 倍放大裁脸，然后 resize 到 `299 x 299`。

## 5. 数据增强

本方案的主要提升来自 `--dfgc-augment`。该增强参考 DFGC 原始比赛中常见的后处理和扰动特征，重点模拟压缩、模糊和低分辨率退化。

训练增强顺序如下：

1. `Resize((299, 299))`
2. `RandomHorizontalFlip()`
3. `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)`
4. 以 `p=0.9` 从以下三种退化中随机选择一种：
   - Gaussian blur，半径范围 `0.4 - 1.6`
   - JPEG compression，质量范围 `35 - 80`
   - Downscale，再 resize 回原尺寸，缩放比例 `0.25 - 0.6`
5. `ToTensor()`
6. 以 `p=0.15` 加入轻微 tensor noise，`sigma=0.02`
7. `Normalize([0.5]*3, [0.5]*3)`

这些增强的目的不是单纯提升 Celeb 内部验证集表现，而是让模型减少对 Celeb 原始视频纹理的过拟合，增强对比赛测试图像中压缩、模糊、降采样、扰动等域偏移的鲁棒性。

## 6. 模型结构

模型使用官方 baseline 的 Xception：

```python
model = Xception()
model.fc = nn.Linear(2048, 1)
```

初始化权重来自官方 baseline：

`/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/utils/weights.ckpt`

训练时模型输出 logit，使用 `BCEWithLogitsLoss` 进行二分类训练。推理时 starterkit 的 `Xception.forward()` 输出 sigmoid 后的 fake probability，分数范围为 `[0, 1]`，其中：

| 分数含义 | 标签 |
|---|---:|
| real | 0 |
| fake | 1 |

## 7. 微调策略

为了避免在 Celeb 上过拟合过强，训练采用保守微调策略：

1. 冻结大部分 Xception backbone。
2. 只解冻最后部分高层特征和分类头：
   - `block12`
   - `conv3`
   - `bn3`
   - `conv4`
   - `bn4`
   - `fc`
3. 使用很小学习率 `3e-6`。
4. 使用 label smoothing `0.05`。
5. 不使用额外模型融合或额外 checkpoint，只训练并提交单个 Xception 模型。

最终可训练参数量为：

| trainable params | total params |
|---:|---:|
| 6,790,433 | 20,809,001 |

这个设置的目标是保留官方 baseline 已经学到的通用伪造检测特征，只让模型高层适配 Celeb 数据和 DFGC 风格退化增强。

## 8. 训练配置

最终训练命令位于：

`/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/train_celeb_fair_xception_dfgcaug.sh`

主要配置如下：

| 参数 | 值 |
|---|---|
| GPU | `CUDA_VISIBLE_DEVICES=2,3,4,5` |
| 多卡方式 | `torch.nn.DataParallel` |
| epoch | 4 |
| batch size | 32 |
| frames per video | 6 |
| learning rate | `3e-6` |
| optimizer | AdamW |
| weight decay | `1e-4` |
| num workers | 12 |
| class sampling | WeightedRandomSampler |
| label smoothing | 0.05 |
| input size | 299 x 299 |
| checkpoint selection | Celeb 内部验证 AUC 最优 |

训练命令如下：

```bash
sh train_celeb_fair_xception_dfgcaug.sh
```

训练过程中保存：

```text
checkpoints/baseline_ucf_celeb_fair_xception_dfgcaug/last.ckpt
checkpoints/baseline_ucf_celeb_fair_xception_dfgcaug/best_auc.ckpt
```

最终使用 `best_auc.ckpt` 进行测试集推理。

## 9. 评估方法

课程验证集只用于训练完成后的离线评估：

`/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/val/val_gts.xlsx`

评估指标包括：

| 指标 | 含义 |
|---|---|
| AUC | 衡量模型对 real/fake 排序能力，阈值无关 |
| AP | Average Precision，关注 fake 类排序质量 |
| acc@0.5 | 以 0.5 为阈值计算准确率 |
| mean_score | 所有样本 fake probability 平均值 |

其中 AUC 是本次优化中最主要的参考指标。

## 10. 实验结果

不同方案在课程验证集上的结果如下：

| 方案 | AUC | AP | acc@0.5 | mean_score |
|---|---:|---:|---:|---:|
| 官方 baseline | 0.635140 | 0.953467 | 0.149389 | 0.156847 |
| Celeb 保守 fc-only 微调 | 0.635169 | - | - | - |
| Celeb fair Xception | 0.695399 | 0.964707 | 0.182500 | 0.174510 |
| Celeb fair Xception + no-hflip | 0.689621 | 0.963655 | 0.183167 | 0.179217 |
| Celeb fair Xception + hflip TTA | 0.678171 | 0.964015 | 0.176667 | 0.172942 |
| 最终方案：Celeb fair Xception + DFGC augment | 0.744581 | 0.970333 | 0.231444 | 0.233098 |

最终方案相较官方 baseline：

```text
AUC: 0.635140 -> 0.744581
提升: +0.109441
```

相较未加入 DFGC 退化增强的 Celeb fair Xception：

```text
AUC: 0.695399 -> 0.744581
提升: +0.049182
```

额外实验显示，最终 DFGC augment 单模型已经优于与官方 baseline 或原 fair Xception 进行线性融合的结果，因此最终提交采用单模型结果。

## 11. 测试集推理与提交文件

使用以下命令对 test1 生成提交结果：

```bash
CUDA_VISIBLE_DEVICES=2 /home/duyijie/.conda/envs/course_AISA/bin/python inference.py \
    --your-team-name Hardworking111_dfgcaug \
    --data-folder /home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/test1 \
    --model-weights /home/duyijie/DeepfakesAdvTrack-Spring2026/detection/checkpoints/baseline_ucf_celeb_fair_xception_dfgcaug/best_auc.ckpt \
    --result-path /home/duyijie/DeepfakesAdvTrack-Spring2026/detection/results
```

生成文件：

`/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/results/Hardworking111_dfgcaug.xlsx`

该文件包含 `20000` 条预测。由于 starterkit 原始 `FolderDataset.get_img_name()` 会保留 `img_list.txt` 行尾换行符，我们在生成后对 `img_names` 做了 `strip()` 清理，以保证提交文件中的图像名格式更干净。

## 12. 方法总结

本方案的关键不是大幅改变模型结构，而是在公平使用 Celeb 训练数据的前提下，通过保守微调和针对性数据增强提升跨域泛化能力。

主要有效因素包括：

1. 使用官方 Xception baseline 权重初始化，保留已有伪造检测能力。
2. 只微调高层模块和分类头，避免低层特征在 Celeb 上过拟合。
3. 使用类别均衡采样，缓解 real/fake 数量不均衡。
4. 使用 label smoothing，提高泛化稳定性。
5. 使用 DFGC 风格退化增强，显著提升对比赛验证集的 AUC。

最终结果表明，压缩、模糊、降采样等退化增强对本任务的跨域表现非常关键，是本次从 `0.695399` 提升到 `0.744581` 的主要原因。
