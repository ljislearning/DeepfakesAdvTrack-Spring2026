import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from hardworking111_data import CroppedFaceTrainSet, build_eval_transform, build_train_transform
from hardworking111_model import Hardworking111TriGuard


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-dir", default="checkpoints/hardworking111_triguard")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--artifact-copies", type=int, default=2)
    parser.add_argument("--init-weights", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--multi-gpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_loss = 0.0
    for images, target in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        target = target.to(device)
        logits = model(images)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / max(1, len(loader.dataset))


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    labels, scores = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, target in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            total_loss += criterion(logits, target).item() * images.size(0)
            score = Hardworking111TriGuard.fake_score_from_logits(logits)
            labels.extend((target > 0).long().cpu().numpy().tolist())
            scores.extend(score.cpu().numpy().tolist())

    preds = [1 if score >= 0.5 else 0 for score in scores]
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "auc": metrics.roc_auc_score(labels, scores),
        "ap": metrics.average_precision_score(labels, scores),
        "acc@0.5": metrics.accuracy_score(labels, preds),
        "mean_score": float(np.mean(scores)),
    }


def unwrap_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.model.state_dict()
    return model.model.state_dict()


def main():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_set = CroppedFaceTrainSet(args.train_csv, build_train_transform(), artifact_copies=args.artifact_copies)
    val_set = CroppedFaceTrainSet(args.val_csv, build_eval_transform(), artifact_copies=0)
    labels = torch.tensor(train_set.get_labels(), dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=3).float().clamp_min(1.0)
    weights = (1.0 / class_counts)[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Hardworking111TriGuard(args.init_weights).to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    best_auc = -1.0
    history = []
    print("train={} val={} class_counts={}".format(len(train_set), len(val_set), class_counts.tolist()))
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "lr": scheduler.get_last_lr()[0], "seconds": time.time() - start}
        row.update(val_metrics)
        history.append(row)

        torch.save(unwrap_state_dict(model), os.path.join(args.output_dir, "last.pth"))
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(unwrap_state_dict(model), os.path.join(args.output_dir, "best_auc.pth"))
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    with open(os.path.join(args.output_dir, "history.json"), "w") as writer:
        json.dump(history, writer, indent=2)
    print("saved best checkpoint:", os.path.join(args.output_dir, "best_auc.pth"))


if __name__ == "__main__":
    main()
