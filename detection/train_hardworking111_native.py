import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset.dataset import DFGCDataset
from loss.losses import LabelSmoothing
from network.models import model_selection


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-root", required=True)
    parser.add_argument("--output-dir", default="checkpoints/hardworking111_native")
    parser.add_argument("--epochs", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--no-adv", action="store_true")
    parser.add_argument("--no-blending", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def labels_to_weights(labels):
    labels = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float().clamp_min(1.0)
    return (1.0 / counts)[labels]


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total = 0
    for images, target in tqdm(loader, desc="train epoch {}".format(epoch), leave=False):
        images = images.to(device)
        target = target.to(device)
        logits = model(images)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / max(1, total)


def evaluate(model, loader, device, epoch):
    model.eval()
    labels = []
    scores = []
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, target in tqdm(loader, desc="val epoch {}".format(epoch), leave=False):
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            total_loss += criterion(logits, target).item() * images.size(0)
            probs = torch.softmax(logits, dim=1)
            score = 1.0 - probs[:, 0]
            binary_target = target.clone()
            binary_target[binary_target > 0] = 1
            labels.extend(binary_target.cpu().numpy().tolist())
            scores.extend(score.cpu().numpy().tolist())
            total += images.size(0)

    preds = [1 if x >= 0.5 else 0 for x in scores]
    return {
        "loss": total_loss / max(1, total),
        "auc": roc_auc_score(labels, scores),
        "ap": average_precision_score(labels, scores),
        "acc@0.5": accuracy_score(labels, preds),
        "mean_score": float(np.mean(scores)),
    }


def main():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, *_ = model_selection(modelname="efficientnet-b3", num_out_classes=3)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    model = model.to(device)

    train_set = DFGCDataset(
        args.face_root,
        data_type="train",
        is_one_hot=True,
        input_size=args.input_size,
        use_adv=not args.no_adv,
        use_real_adv=False,
        use_blending=not args.no_blending,
        seed=args.seed,
        num_classes=3,
    )
    val_set = DFGCDataset(
        args.face_root,
        data_type="val",
        is_one_hot=False,
        input_size=args.input_size,
        use_adv=False,
        use_blending=False,
        seed=args.seed,
        num_classes=3,
    )

    sampler = WeightedRandomSampler(labels_to_weights(train_set.get_labels()), len(train_set), replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    criterion = LabelSmoothing(smoothing=0.05).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_auc = -1.0
    history = []
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device, epoch)
            row.update(metrics)
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_auc.pth"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pth"))
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    with open(os.path.join(args.output_dir, "history.json"), "w") as writer:
        json.dump(history, writer, indent=2)
    print("saved best checkpoint:", os.path.join(args.output_dir, "best_auc.pth"))


if __name__ == "__main__":
    main()
