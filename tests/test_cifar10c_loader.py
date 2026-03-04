from pathlib import Path

import numpy as np

from rtda.data.cifar import CIFAR10CDataset


def test_cifar10c_slice(tmp_path: Path):
    root = tmp_path / "cifar10c"
    root.mkdir(parents=True)

    imgs = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(50000, dtype=np.int64) % 10
    np.save(root / "gaussian_noise.npy", imgs)
    np.save(root / "labels.npy", labels)

    ds = CIFAR10CDataset(root=root, corruption="gaussian_noise", severity=3)
    assert len(ds) == 10000
    _, y0 = ds[0]
    assert isinstance(y0, int)
