import argparse
import os
import random
import time

import cv2
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFilter
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from utils import Xception


REAL_DIRS = {"YouTube-real", "Celeb-real"}
FAKE_DIRS = {"Celeb-synthesis"}


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/Celeb")
    parser.add_argument("--init-weights", default="utils/weights.ckpt")
    parser.add_argument("--output-dir", default="checkpoints/baseline_ucf_celeb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--no-ucf-augment", action="store_true")
    parser.add_argument("--no-face-crop", action="store_true")
    parser.add_argument("--face-scale", type=float, default=1.3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-block", action="store_true")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--strong-augment", action="store_true")
    parser.add_argument("--dfgc-augment", action="store_true")
    parser.add_argument("--no-hflip", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_videos(data_root):
    samples = []
    for dirname in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, dirname)
        if not os.path.isdir(folder):
            continue
        if dirname in REAL_DIRS:
            label = 0
        elif dirname in FAKE_DIRS:
            label = 1
        else:
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                samples.append((os.path.join(folder, name), label))
    if not samples:
        raise RuntimeError("No videos found under {}".format(data_root))
    return samples


def split_samples(samples, val_ratio, seed):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for item in samples:
        by_label[item[1]].append(item)
    train, val = [], []
    for label_items in by_label.values():
        rng.shuffle(label_items)
        val_count = max(1, int(len(label_items) * val_ratio))
        val.extend(label_items[:val_count])
        train.extend(label_items[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class CelebVideoDataset(Dataset):
    def __init__(
        self,
        samples,
        frames_per_video,
        transform,
        random_frame=True,
        face_crop=True,
        face_scale=1.3,
    ):
        self.items = []
        self.frames_per_video = frames_per_video
        for video_path, label in samples:
            for frame_slot in range(frames_per_video):
                self.items.append((video_path, label, frame_slot))
        self.transform = transform
        self.random_frame = random_frame
        self.face_crop = face_crop
        self.face_scale = face_scale
        self.face_detector = None

    def _get_face_detector(self):
        if self.face_detector is None:
            cascade_path = os.path.join(
                cv2.data.haarcascades,
                "haarcascade_frontalface_default.xml",
            )
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            if self.face_detector.empty():
                raise RuntimeError("Cannot load OpenCV face detector: {}".format(cascade_path))
        return self.face_detector

    def __len__(self):
        return len(self.items)

    def _read_frame(self, video_path, frame_slot):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video: {}".format(video_path))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_idx = 0
        elif self.random_frame:
            frame_idx = random.randint(0, max(0, frame_count - 1))
        else:
            ratio = (frame_slot + 0.5) / float(self.frames_per_video)
            frame_idx = min(frame_count - 1, int(frame_count * ratio))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Cannot read frame from video: {}".format(video_path))
        if self.face_crop:
            frame = self._crop_face(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def _crop_face(self, frame):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._get_face_detector().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            size = int(max(w, h) * self.face_scale)
        else:
            center_x = width / 2.0
            center_y = height / 2.0
            size = min(width, height)

        x1 = max(int(center_x - size / 2.0), 0)
        y1 = max(int(center_y - size / 2.0), 0)
        x2 = min(x1 + size, width)
        y2 = min(y1 + size, height)
        x1 = max(x2 - size, 0)
        y1 = max(y2 - size, 0)
        return frame[y1:y2, x1:x2]

    def __getitem__(self, index):
        video_path, label, frame_slot = self.items[index]
        image = self._read_frame(video_path, frame_slot)
        return self.transform(image), torch.tensor(label, dtype=torch.float32)

    def get_labels(self):
        return [label for _, label, _ in self.items]


class RandomJPEGCompression(object):
    def __init__(self, p=0.2, quality_range=(45, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomPILGaussianBlur(object):
    def __init__(self, p=0.15, radius_range=(0.4, 1.2)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        radius = random.uniform(self.radius_range[0], self.radius_range[1])
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomPILDownscale(object):
    def __init__(self, p=0.2, scale_range=(0.25, 0.6)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        width, height = image.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(small_size, Image.BILINEAR)
        return image.resize((width, height), Image.BILINEAR)


class RandomOneOf(object):
    def __init__(self, transforms_list, p=1.0):
        self.transforms_list = transforms_list
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        transform = random.choice(self.transforms_list)
        return transform(image)


class RandomTensorNoise(object):
    def __init__(self, p=0.15, sigma=0.025):
        self.p = p
        self.sigma = sigma

    def __call__(self, tensor):
        if random.random() >= self.p:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.sigma, 0.0, 1.0)


class BaselineXceptionWithUcfAugment(nn.Module):
    def __init__(self, init_weights=None, freeze_backbone=False, unfreeze_last_block=False):
        super().__init__()
        self.model = Xception()
        self.model.fc = nn.Linear(2048, 1)
        if init_weights:
            state_dict = torch.load(init_weights, map_location="cpu")
            self.model.load_state_dict(state_dict)
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_block = unfreeze_last_block
        if freeze_backbone:
            self.freeze_for_finetune(unfreeze_last_block)

    def freeze_for_finetune(self, unfreeze_last_block=False):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_modules = [self.model.fc]
        if unfreeze_last_block:
            trainable_modules.extend(
                [
                    self.model.block12,
                    self.model.conv3,
                    self.model.bn3,
                    self.model.conv4,
                    self.model.bn4,
                ]
            )

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

    def keep_frozen_modules_eval(self):
        if not self.freeze_backbone:
            return
        self.model.eval()
        if self.unfreeze_last_block:
            self.model.block12.train()
            self.model.conv3.train()
            self.model.bn3.train()
            self.model.conv4.train()
            self.model.bn4.train()
        self.model.fc.train()

    def _augment_features(self, features):
        # Lightweight version of the UCF latent perturbation/mixup idea.
        bs = features.size(0)
        augmented = features

        if bs > 1:
            flat = features.view(bs, -1)
            mean = flat.mean(dim=0, keepdim=True)
            distances = torch.norm(flat - mean, dim=1)
            hard = features[torch.argmax(distances)].unsqueeze(0)
            lam = torch.rand(bs, 1, 1, 1, device=features.device) * 0.5
            augmented = features + lam * (hard - features)

        if random.random() < 0.5 and bs > 1:
            alpha = 0.4 + 1.6 * random.random()
            lam = torch.distributions.Beta(alpha, alpha).sample((bs,)).to(features.device)
            lam = lam.view(bs, 1, 1, 1)
            shuffled = augmented[torch.randperm(bs, device=features.device)]
            augmented = lam * augmented + (1 - lam) * shuffled

        if random.random() < 0.3:
            sigma = torch.rand(1, device=features.device) * 0.05
            augmented = augmented + torch.randn_like(augmented) * sigma

        return augmented

    def forward_logits(self, images, use_ucf_augment=False):
        features = self.model.features(images)
        if self.training and use_ucf_augment:
            features = self._augment_features(features)
        logits = self.model.logits(features)
        return logits.view(-1)

    def forward(self, images, use_ucf_augment=False):
        return self.forward_logits(images, use_ucf_augment=use_ucf_augment)

    def export_state_dict(self):
        return self.model.state_dict()


def build_transforms(train, strong_augment=False, hflip=True, dfgc_augment=False):
    if train:
        transform_list = [
            transforms.Resize((299, 299)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ]
        if hflip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())
        if strong_augment:
            transform_list.extend(
                [
                    RandomPILGaussianBlur(p=0.15),
                    RandomJPEGCompression(p=0.25, quality_range=(45, 95)),
                ]
            )
        if dfgc_augment:
            transform_list.append(
                RandomOneOf(
                    [
                        RandomPILGaussianBlur(p=1.0, radius_range=(0.4, 1.6)),
                        RandomJPEGCompression(p=1.0, quality_range=(35, 80)),
                        RandomPILDownscale(p=1.0, scale_range=(0.25, 0.6)),
                    ],
                    p=0.9,
                )
            )
        transform_list.append(transforms.ToTensor())
        if strong_augment or dfgc_augment:
            transform_list.append(RandomTensorNoise(p=0.15, sigma=0.02))
        transform_list.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
        return transforms.Compose(transform_list)
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, target in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            target = target.to(device)
            logits = forward_logits(model, images, use_ucf_augment=False)
            loss = criterion(logits, target)
            score = torch.sigmoid(logits)
            total_loss += loss.item() * images.size(0)
            labels.extend(target.cpu().numpy().tolist())
            scores.extend(score.cpu().numpy().tolist())

    pred = [1 if score >= 0.5 else 0 for score in scores]
    auc = metrics.roc_auc_score(labels, scores) if len(set(labels)) == 2 else 0.0
    ap = metrics.average_precision_score(labels, scores) if len(set(labels)) == 2 else 0.0
    acc = metrics.accuracy_score(labels, pred)
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "auc": auc,
        "ap": ap,
        "acc": acc,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
    }


def smooth_targets(target, smoothing):
    if smoothing <= 0:
        return target
    return target * (1.0 - smoothing) + 0.5 * smoothing


def train_one_epoch(model, loader, optimizer, device, use_ucf_augment, label_smoothing):
    model.train()
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.keep_frozen_modules_eval()
    criterion = nn.BCEWithLogitsLoss()
    running = 0.0
    for images, target in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        target = target.to(device)
        smooth_target = smooth_targets(target, label_smoothing)
        logits = forward_logits(model, images, use_ucf_augment=use_ucf_augment)
        loss = criterion(logits, smooth_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / max(1, len(loader.dataset))


def forward_logits(model, images, use_ucf_augment):
    return model(images, use_ucf_augment=use_ucf_augment)


def export_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.export_state_dict()
    return model.export_state_dict()


def main():
    opts = get_opts()
    seed_everything(opts.seed)
    os.makedirs(opts.output_dir, exist_ok=True)

    samples = collect_videos(opts.data_root)
    train_samples, val_samples = split_samples(samples, opts.val_ratio, opts.seed)
    print("videos: total={} train={} val={}".format(len(samples), len(train_samples), len(val_samples)))
    print(
        "train real/fake={}/{} val real/fake={}/{}".format(
            sum(1 for _, label in train_samples if label == 0),
            sum(1 for _, label in train_samples if label == 1),
            sum(1 for _, label in val_samples if label == 0),
            sum(1 for _, label in val_samples if label == 1),
        )
    )

    train_dataset = CelebVideoDataset(
        train_samples,
        frames_per_video=opts.frames_per_video,
        transform=build_transforms(
            train=True,
            strong_augment=opts.strong_augment,
            hflip=not opts.no_hflip,
            dfgc_augment=opts.dfgc_augment,
        ),
        random_frame=True,
        face_crop=not opts.no_face_crop,
        face_scale=opts.face_scale,
    )
    val_dataset = CelebVideoDataset(
        val_samples,
        frames_per_video=max(1, opts.frames_per_video),
        transform=build_transforms(train=False),
        random_frame=False,
        face_crop=not opts.no_face_crop,
        face_scale=opts.face_scale,
    )

    train_sampler = None
    train_shuffle = True
    if opts.balanced_sampler:
        train_labels = torch.tensor(train_dataset.get_labels(), dtype=torch.long)
        class_counts = torch.bincount(train_labels)
        class_weights = 1.0 / class_counts.float().clamp_min(1.0)
        sample_weights = class_weights[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
        print(
            "balanced sampler class_counts={} class_weights={}".format(
                class_counts.tolist(),
                ["{:.6f}".format(float(x)) for x in class_weights],
            )
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
    )

    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")
    model = BaselineXceptionWithUcfAugment(
        opts.init_weights,
        freeze_backbone=opts.freeze_backbone,
        unfreeze_last_block=opts.unfreeze_last_block,
    ).to(device)
    if opts.multi_gpu and torch.cuda.device_count() > 1:
        print("using {} GPUs with DataParallel".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    print(
        "trainable params: {} / {}".format(
            sum(param.numel() for param in trainable_params),
            sum(param.numel() for param in model.parameters()),
        )
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=opts.lr, weight_decay=opts.weight_decay)

    best_auc = -1.0
    best_path = os.path.join(opts.output_dir, "best_auc.ckpt")
    last_path = os.path.join(opts.output_dir, "last.ckpt")
    use_ucf_augment = not opts.no_ucf_augment

    for epoch in range(1, opts.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_ucf_augment,
            opts.label_smoothing,
        )
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        torch.save(export_state_dict(model), last_path)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(export_state_dict(model), best_path)

        print(
            "epoch {}/{} train_loss={:.6f} val_loss={:.6f} auc={:.6f} ap={:.6f} "
            "acc@0.5={:.6f} mean_score={:.6f} time={:.1f}s best_auc={:.6f}".format(
                epoch,
                opts.epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["auc"],
                val_metrics["ap"],
                val_metrics["acc"],
                val_metrics["mean_score"],
                elapsed,
                best_auc,
            )
        )

    print("saved best checkpoint: {}".format(best_path))


if __name__ == "__main__":
    main()
