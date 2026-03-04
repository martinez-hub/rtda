from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from rtda.augmentations.ops import get_augmentations


class AugMix:
    def __init__(
        self,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        preprocess: transforms.Compose | None = None,
    ) -> None:
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = get_augmentations(all_ops=all_ops)
        self.preprocess = preprocess or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = torch.zeros_like(self.preprocess(image))
        for i in range(self.width):
            image_aug = image.copy()
            d = self.depth if self.depth > 0 else random.randint(1, 3)
            for _ in range(d):
                op = random.choice(self.augmentations)
                image_aug = op(image_aug, self.severity)
            mix = mix + ws[i] * self.preprocess(image_aug)

        clean = self.preprocess(image)
        return (1 - m) * clean + m * mix
