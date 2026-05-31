import argparse
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import Xception


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="dataset/val")
    parser.add_argument("--gt-path", default="dataset/val/val_gts.xlsx")
    parser.add_argument("--init-weights", default="utils/weights.ckpt")
    parser.add_argument("--output-dir", default="checkpoints/val_sanity")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--multi-gpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_name(name):
    return os.path.splitext(str(name).strip())[0]


class LabeledFolderDataset(Dataset):
    def __init__(self, folder, gt_path, transform, scale=1.3):
        self.folder = folder
        self.img_list = [line.strip() for line in open(os.path.join(folder, "img_list.txt"), "r")]
        self.face_info = [line.strip() for line in open(os.path.join(folder, "face_info.txt"), "r")]
        if len(self.img_list) != len(self.face_info):
            raise RuntimeError("img_list and face_info length mismatch")

        gt = pd.read_excel(gt_path)
        gt["key"] = gt["img_names"].map(normalize_name)
        labels = dict(zip(gt["key"], gt["labels"]))

        self.labels = []
        missing = []
        for name in self.img_list:
            key = normalize_name(name)
            if key not in labels:
                missing.append(name)
                self.labels.append(0)
            else:
                self.labels.append(float(labels[key]))
        if missing:
            raise RuntimeError("Missing labels for first samples: {}".format(missing[:10]))

        self.transform = transform
        self.scale = scale

    def __len__(self):
        return len(self.img_list)

    def read_crop_face(self, idx):
        img = cv2.imread(os.path.join(self.folder, "imgs", self.img_list[idx]))
        if img is None:
            raise RuntimeError("Cannot read image: {}".format(self.img_list[idx]))
        height, width = img.shape[:2]

        box = [float(x) for x in self.face_info[idx].split(" ")]
        x1, y1, x2, y2 = box[:4]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        size_bb = int(max(x2 - x1, y2 - y1) * self.scale)
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)
        return img[y1 : y1 + size_bb, x1 : x1 + size_bb]

    def __getitem__(self, idx):
        img = self.read_crop_face(idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def build_model(init_weights):
    model = Xception()
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(init_weights, map_location="cpu"))
    return model


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, target in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            target = target.to(device)
            logits = model(images).view(-1)
            loss = criterion(logits, target)
            score = torch.sigmoid(logits)
            total_loss += loss.item() * images.size(0)
            labels.extend(target.cpu().numpy().tolist())
            scores.extend(score.cpu().numpy().tolist())

    pred = (np.array(scores) >= 0.5).astype(np.int64)
    labels_np = np.array(labels).astype(np.int64)
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "auc": metrics.roc_auc_score(labels_np, scores),
        "ap": metrics.average_precision_score(labels_np, scores),
        "acc": metrics.accuracy_score(labels_np, pred),
        "mean_score": float(np.mean(scores)),
    }


def train_one_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    running = 0.0
    for images, target in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        target = target.to(device)
        logits = model(images).view(-1)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / max(1, len(loader.dataset))


def main():
    opts = get_opts()
    seed_everything(opts.seed)
    os.makedirs(opts.output_dir, exist_ok=True)

    dataset = LabeledFolderDataset(opts.data_folder, opts.gt_path, build_transform())
    labels = np.array(dataset.labels)
    real_count = int((labels == 0).sum())
    fake_count = int((labels == 1).sum())
    print("samples={} real={} fake={}".format(len(dataset), real_count, fake_count))

    loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
    )

    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")
    model = build_model(opts.init_weights).to(device)
    if opts.multi_gpu and torch.cuda.device_count() > 1:
        print("using {} GPUs with DataParallel".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    pos_weight = torch.tensor([real_count / float(max(1, fake_count))], device=device)
    print("pos_weight={:.6f}".format(float(pos_weight.item())))

    before = evaluate(model, eval_loader, device)
    print(
        "before train loss={loss:.6f} auc={auc:.6f} ap={ap:.6f} "
        "acc@0.5={acc:.6f} mean_score={mean_score:.6f}".format(**before)
    )

    best_auc = before["auc"]
    best_path = os.path.join(opts.output_dir, "best_auc.ckpt")
    torch.save(unwrap(model).state_dict(), best_path)

    for epoch in range(1, opts.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, eval_loader, device)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(unwrap(model).state_dict(), best_path)
        print(
            "epoch {}/{} train_loss={:.6f} eval_loss={:.6f} auc={:.6f} ap={:.6f} "
            "acc@0.5={:.6f} mean_score={:.6f} time={:.1f}s best_auc={:.6f}".format(
                epoch,
                opts.epochs,
                train_loss,
                val_metrics["loss"],
                val_metrics["auc"],
                val_metrics["ap"],
                val_metrics["acc"],
                val_metrics["mean_score"],
                time.time() - start,
                best_auc,
            )
        )

    print("saved best checkpoint: {}".format(best_path))


if __name__ == "__main__":
    main()
