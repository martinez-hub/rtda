from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, wide_resnet50_2


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = resnet18(num_classes=num_classes)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


class CIFARWideResNet50_2(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = wide_resnet50_2(weights=None, num_classes=num_classes)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


def build_model(cfg: dict) -> nn.Module:
    model_name = cfg.get("model", {}).get("name", "wrn50_2")
    if model_name == "wrn50_2":
        return CIFARWideResNet50_2(num_classes=10)
    if model_name == "resnet18":
        return CIFARResNet18(num_classes=10)
    raise ValueError(f"Unsupported model.name: {model_name}")
