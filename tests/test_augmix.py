import numpy as np
from PIL import Image

from rtda.augmentations.augmix import AugMix


def test_augmix_tensor_shape_and_dtype():
    aug = AugMix(severity=1, width=2, depth=1, alpha=1.0)
    arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    out = aug(img)
    assert tuple(out.shape) == (3, 32, 32)
    assert str(out.dtype) == "torch.float32"
