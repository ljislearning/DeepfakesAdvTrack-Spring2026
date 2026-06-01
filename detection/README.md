# Hardworking111 TriGuard Detection

`detection_new` is the clean submission folder for the Hardworking111 deepfake detector. It contains one solution only: **Hardworking111 TriGuard**.

## Folder Layout

```text
detection_new/
  data_preparation/                face extraction, landmark extraction, adversarial image generation
  models/                         local EfficientNet-B3 implementation
  weights/hardworking111_triguard_b3.pth
  dataset/                         native face dataset loader and blending logic
  loss/                            label smoothing loss
  network/                         EfficientNet-B3 training model
  utils/                           training utilities
  prepare_celeb_faces.py           optional Celeb-DF frame extraction and face cropping
  train_hardworking111_native.py   native training entry
  infer_hardworking111.py          inference entry
  calibrate_hardworking111.py      score calibration entry
  scripts/01_extract_faces.sh       video to face crops
  scripts/02_generate_landmarks.sh  face landmarks for blending
  scripts/03_generate_adversarial.sh fake artifact subset generation
  scripts/04_train_detector.sh      detector training
  scripts/05_infer_test1.sh         test1 inference and calibration
  scripts/run_full_pipeline.sh      full preprocessing-training-inference pipeline
  TECH_REPORT.md                   method report
```

## Full Pipeline

The complete pipeline is:

```text
raw Celeb videos
  -> extracted face images
  -> dlib landmark json files
  -> adversarial/artifact fake subsets
  -> three-class EfficientNet-B3 training
  -> test inference
  -> score calibration
```

Run each stage explicitly:

```bash
cd /home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new

VIDEO_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection/dataset/Celeb \
FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
PYTHON=/path/to/preprocess_env/bin/python \
GPU=0 \
sh scripts/01_extract_faces.sh

FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
PYTHON=/path/to/preprocess_env/bin/python \
sh scripts/02_generate_landmarks.sh

FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
PYTHON=/path/to/attack_env/bin/python \
GPU=0 \
sh scripts/03_generate_adversarial.sh

FACE_ROOT=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/data/Celeb-DF-v2-face \
sh scripts/04_train_detector.sh

WEIGHTS=/home/duyijie/DeepfakesAdvTrack-Spring2026/detection_new/checkpoints/hardworking111_native/best_auc.pth \
sh scripts/05_infer_test1.sh
```

Or run all stages:

```bash
sh scripts/run_full_pipeline.sh
```

## Required Data And Weights

Recommended local environments:

```text
preprocessing/training: /home/duyijie/.conda/envs/DeepfakeBench/bin/python
inference/xlsx output: /home/duyijie/.conda/envs/course_AISA/bin/python
```

The `DeepfakeBench` environment has been prepared with the preprocessing dependencies used by this folder, including `dlib`, `facenet-pytorch`, `albumentations`, and `efficientnet_pytorch`.

The native training path expects a face-root directory with extracted face folders and optional artifact folders:

```text
Celeb-real/
YouTube-real/
Celeb-synthesis/
Celeb-synthesis_adv1/
Celeb-synthesis_adv2/
Celeb-synthesis_adv3/
Celeb-synthesis_adv4/
Celeb-synthesis_adv5/
```

`scripts/03_generate_adversarial.sh` requires attack baseline weights under:

```text
data_preparation/generate_adversarial/weights/
```

If those weights are absent, train them with the scripts under `data_preparation/train_baseline_model/` or place the downloaded weights in that folder.

## Fast Inference With Provided Detector Weight

The folder also contains the detector weight already available in the workspace:

```bash
sh run_infer_test1.sh
```

The calibrated file will be:

```text
results/Hardworking111_calibrated.xlsx
```
