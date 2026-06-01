import csv
import io
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def jpeg_roundtrip(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class RandomArtifact(object):
    """Synthetic artifact transform used to build the third training class."""

    def __call__(self, image):
        choice = random.randrange(5)
        if choice == 0:
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.35, 1.60)))
        if choice == 1:
            width, height = image.size
            scale = random.uniform(0.35, 0.75)
            small = image.resize((max(2, int(width * scale)), max(2, int(height * scale))), Image.BILINEAR)
            return small.resize((width, height), Image.BILINEAR)
        if choice == 2:
            return jpeg_roundtrip(image, random.randint(42, 84))
        if choice == 3:
            arr = np.asarray(image).astype(np.float32)
            noise = np.random.normal(0.0, random.uniform(2.0, 8.0), arr.shape)
            return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

        arr = np.asarray(image).astype(np.float32)
        scale = random.uniform(0.86, 1.14)
        bias = random.uniform(-8.0, 8.0)
        return Image.fromarray(np.clip(arr * scale + bias, 0, 255).astype(np.uint8))


class TensorNoise(object):
    def __init__(self, p=0.12, sigma=0.010):
        self.p = p
        self.sigma = sigma

    def __call__(self, tensor):
        if random.random() >= self.p:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.sigma, 0.0, 1.0)


def build_train_transform(size=300):
    return transforms.Compose(
        [
            transforms.Resize((size + 24, size + 24)),
            transforms.RandomCrop((size, size)),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
            transforms.RandomApply([RandomArtifact()], p=0.35),
            transforms.ToTensor(),
            TensorNoise(p=0.12, sigma=0.010),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(size=300):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CroppedFaceTrainSet(Dataset):
    def __init__(self, csv_path, transform, artifact_copies=2):
        self.items = []
        with open(csv_path, "r") as reader:
            for row in csv.DictReader(reader):
                label = int(row["label"])
                path = row["path"]
                if label == 0:
                    self.items.append((path, 0, False))
                else:
                    self.items.append((path, 1, False))
                    for _ in range(artifact_copies):
                        self.items.append((path, 2, True))
        if not self.items:
            raise RuntimeError("No samples found in {}".format(csv_path))
        self.transform = transform
        self.artifact = RandomArtifact()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label, use_artifact = self.items[index]
        image = Image.open(path).convert("RGB")
        if use_artifact:
            image = self.artifact(image)
        return self.transform(image), torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [label for _, label, _ in self.items]


class CourseImageFolder(Dataset):
    def __init__(self, folder, transform, face_scale=1.3):
        self.folder = folder
        self.transform = transform
        self.face_scale = face_scale
        with open(os.path.join(folder, "img_list.txt"), "r") as reader:
            self.img_list = [line.strip() for line in reader if line.strip()]
        with open(os.path.join(folder, "face_info.txt"), "r") as reader:
            self.face_info = [line.strip() for line in reader if line.strip()]
        if len(self.img_list) != len(self.face_info):
            raise RuntimeError("img_list and face_info length mismatch in {}".format(folder))

    def __len__(self):
        return len(self.img_list)

    def get_img_name(self):
        return list(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder, "imgs", self.img_list[index])
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError("Cannot read image: {}".format(img_path))

        height, width = img.shape[:2]
        x1, y1, x2, y2 = [float(x) for x in self.face_info[index].split()]
        box_size = int(max(x2 - x1, y2 - y1) * self.face_scale)
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        crop_x1 = max(int(center_x - box_size / 2.0), 0)
        crop_y1 = max(int(center_y - box_size / 2.0), 0)
        box_size = min(box_size, width - crop_x1, height - crop_y1)
        crop = img[crop_y1 : crop_y1 + box_size, crop_x1 : crop_x1 + box_size]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(crop))
