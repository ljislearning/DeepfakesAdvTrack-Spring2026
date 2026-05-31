import importlib.util
import os
import sys

import torch
import torch.nn as nn


def _load_bench_xception(deepfakebench_root):
    training_dir = os.path.join(deepfakebench_root, "training")
    xception_path = os.path.join(training_dir, "networks", "xception.py")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)

    spec = importlib.util.spec_from_file_location("deepfakebench_xception", xception_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Xception


def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def c_norm(self, x, bs, ch, eps=1e-7):
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out = ((x - x_mean.expand(size)) / x_std.expand(size)) * y_std.expand(size) + y_mean.expand(size)
        return out


class Conditional_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dropout = nn.Dropout(p=0.3)
        self.adain3 = AdaIN()
        self.adain2 = AdaIN()
        self.adain1 = AdaIN()
        self.dconv_up3 = r_double_conv(512, 256)
        self.dconv_up2 = r_double_conv(256, 128)
        self.dconv_up1 = r_double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.activation = nn.Tanh()

    def forward(self, c, x):
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up3(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up3(c)

        x = self.adain2(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up2(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up2(c)

        x = self.adain1(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        return self.activation(self.up_last(x))


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_f),
        )

    def forward(self, x):
        bs = x.size(0)
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat


class UCFInferenceModel(nn.Module):
    def __init__(self, deepfakebench_root="/home/duyijie/DeepfakeBench"):
        super().__init__()
        xception_cls = _load_bench_xception(deepfakebench_root)
        backbone_config = {
            "mode": "adjust_channel",
            "num_classes": 2,
            "inc": 3,
            "dropout": False,
        }
        encoder_feat_dim = 512
        half_fingerprint_dim = encoder_feat_dim // 2

        self.encoder_f = xception_cls(backbone_config)
        self.encoder_c = xception_cls(backbone_config)
        self.con_gan = Conditional_UNet()
        self.head_spe = Head(half_fingerprint_dim, encoder_feat_dim, 5)
        self.head_sha = Head(half_fingerprint_dim, encoder_feat_dim, 2)
        self.block_spe = Conv2d1x1(encoder_feat_dim, half_fingerprint_dim, half_fingerprint_dim)
        self.block_sha = Conv2d1x1(encoder_feat_dim, half_fingerprint_dim, half_fingerprint_dim)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
        self.load_state_dict(state_dict, strict=True)

    def forward(self, image):
        forgery_features = self.encoder_f.features(image)
        f_share = self.block_sha(forgery_features)
        out_sha, _ = self.head_sha(f_share)
        return torch.softmax(out_sha, dim=1)[:, 1]
