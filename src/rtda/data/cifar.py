from __future__ import annotations

from functools import partial
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from rtda.augmentations.augmix import AugMix


CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


class RobustAugMixTrainWrapper(Dataset):
    def __init__(self, base_dataset: datasets.CIFAR10, augmix: AugMix, preprocess, preaugment) -> None:
        self.base = base_dataset
        self.augmix = augmix
        self.preprocess = preprocess
        self.preaugment = preaugment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base.data[idx], int(self.base.targets[idx])
        pil = self.preaugment(Image.fromarray(img))
        clean = self.preprocess(pil)
        aug = self.augmix(pil)
        return clean, aug, label


class AugMixTrainWrapper(Dataset):
    def __init__(self, base_dataset: datasets.CIFAR10, augmix: AugMix, preprocess, preaugment) -> None:
        self.base = base_dataset
        self.augmix = augmix
        self.preprocess = preprocess
        self.preaugment = preaugment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base.data[idx], int(self.base.targets[idx])
        pil = self.preaugment(Image.fromarray(img))
        clean = self.preprocess(pil)
        aug1 = self.augmix(pil)
        aug2 = self.augmix(pil)
        return clean, aug1, aug2, label


class CIFAR10CDataset(Dataset):
    def __init__(self, root: str | Path, corruption: str, severity: int, transform=None) -> None:
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        labels_path = self.root / "labels.npy"
        corruption_path = self.root / f"{corruption}.npy"
        if not labels_path.exists() or not corruption_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10-C files in {self.root}")

        self.labels = np.load(labels_path)
        arr = np.load(corruption_path)

        start = (severity - 1) * 10000
        end = severity * 10000
        self.images = arr[start:end]
        self.labels = self.labels[start:end]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def default_preprocess():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def train_preaugment():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]
    )


def train_preprocess():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def build_train_loader(cfg: dict, epoch: int = 0):
    preprocess = default_preprocess()
    preaugment = train_preaugment()
    base = datasets.CIFAR10(root=cfg["dataset"]["data_root"], train=True, download=True, transform=None)

    method = cfg["train"]["method"]
    augmix = AugMix(
        severity=cfg["augment"]["severity"],
        width=cfg["augment"]["augmix_width"],
        depth=cfg["augment"]["augmix_depth"],
        alpha=cfg["augment"]["alpha"],
        all_ops=bool(cfg["augment"].get("all_ops", True)),
        preprocess=preprocess,
    )

    if method in {"vanilla", "adversarial"}:
        base.transform = train_preprocess()
        ds = base
    elif method == "augmix":
        ds = AugMixTrainWrapper(
            base_dataset=base, augmix=augmix, preprocess=preprocess, preaugment=preaugment
        )
    else:
        ds = RobustAugMixTrainWrapper(
            base_dataset=base, augmix=augmix, preprocess=preprocess, preaugment=preaugment
        )

    return DataLoader(
        ds,
        shuffle=True,
        **_loader_kwargs(cfg, batch_size=cfg["train"]["batch_size"], epoch=epoch),
    )


def build_test_loader(cfg: dict):
    ds = datasets.CIFAR10(
        root=cfg["dataset"]["data_root"],
        train=False,
        download=True,
        transform=default_preprocess(),
    )
    return DataLoader(
        ds,
        shuffle=False,
        **_loader_kwargs(cfg, batch_size=cfg["train"]["batch_size"], epoch=0),
    )


def build_cifar10c_loader(cfg: dict, corruption: str, severity: int):
    ds = CIFAR10CDataset(
        root=cfg["dataset"]["cifar10c_root"],
        corruption=corruption,
        severity=severity,
        transform=default_preprocess(),
    )
    return DataLoader(
        ds,
        shuffle=False,
        **_loader_kwargs(cfg, batch_size=cfg["train"]["batch_size"], epoch=0),
    )


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _loader_kwargs(cfg: dict, batch_size: int, epoch: int) -> dict:
    num_workers = int(cfg["system"]["num_workers"])
    base_seed = int(cfg["system"]["seed"]) + int(epoch)
    generator = torch.Generator()
    generator.manual_seed(base_seed)

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
        "worker_init_fn": partial(_seed_worker, base_seed=base_seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg["system"].get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg["system"].get("prefetch_factor", 2))
    return kwargs
