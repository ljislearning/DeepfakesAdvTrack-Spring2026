import os
import sys

import torch
import torch.nn as nn


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(THIS_DIR, "models")
DEFAULT_WEIGHTS = os.path.join(THIS_DIR, "weights", "hardworking111_triguard_b3.pth")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from efficientnet import TransferModel  # noqa: E402


class Hardworking111TriGuard(nn.Module):
    """EfficientNet-B3 three-way detector.

    Class 0 is real, class 1 is clean fake, and class 2 is artifact-augmented fake.
    During inference class 1 and class 2 are merged into a single fake score.
    """

    def __init__(self, weights_path=None):
        super().__init__()
        self.model = TransferModel("efficientnet-b3", num_out_classes=3)
        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)

    def forward(self, images):
        return self.model(images)

    @staticmethod
    def fake_score_from_logits(logits):
        probs = torch.softmax(logits, dim=1)
        return 1.0 - probs[:, 0]

    def predict_fake(self, images, tta=True):
        if not tta:
            return self.fake_score_from_logits(self.forward(images))

        probs = torch.softmax(self.forward(images), dim=1)
        probs = probs + torch.softmax(self.forward(images.flip(dims=(2,))), dim=1)
        probs = probs + torch.softmax(self.forward(images.flip(dims=(3,))), dim=1)
        probs = probs / 3.0
        return 1.0 - probs[:, 0]
