from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


PARAMETER_MAX = 10


def _float_parameter(level: float, maxval: float) -> float:
    return float(level) * maxval / PARAMETER_MAX


def _int_parameter(level: float, maxval: int) -> int:
    return int(level * maxval / PARAMETER_MAX)


def _sample_level(severity: int) -> float:
    return np.random.uniform(low=0.1, high=severity)


def autocontrast(img: Image.Image, _: int) -> Image.Image:
    return ImageOps.autocontrast(img)


def equalize(img: Image.Image, _: int) -> Image.Image:
    return ImageOps.equalize(img)


def posterize(img: Image.Image, level: int) -> Image.Image:
    level = 4 - _int_parameter(_sample_level(level), 4)
    return ImageOps.posterize(img, max(1, level))


def rotate(img: Image.Image, level: int) -> Image.Image:
    degrees = _int_parameter(_sample_level(level), 30)
    if random.random() > 0.5:
        degrees = -degrees
    return img.rotate(degrees)


def solarize(img: Image.Image, level: int) -> Image.Image:
    level = 256 - _int_parameter(_sample_level(level), 256)
    return ImageOps.solarize(img, level)


def shear_x(img: Image.Image, level: int) -> Image.Image:
    level = _float_parameter(_sample_level(level), 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), Image.BICUBIC)


def shear_y(img: Image.Image, level: int) -> Image.Image:
    level = _float_parameter(_sample_level(level), 0.3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), Image.BICUBIC)


def translate_x(img: Image.Image, level: int) -> Image.Image:
    level = _int_parameter(_sample_level(level), img.size[0] / 3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), Image.BICUBIC)


def translate_y(img: Image.Image, level: int) -> Image.Image:
    level = _int_parameter(_sample_level(level), img.size[1] / 3)
    if random.random() > 0.5:
        level = -level
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), Image.BICUBIC)


def color(img: Image.Image, level: int) -> Image.Image:
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(img).enhance(level)


def contrast(img: Image.Image, level: int) -> Image.Image:
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(img).enhance(level)


def brightness(img: Image.Image, level: int) -> Image.Image:
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(img).enhance(level)


def sharpness(img: Image.Image, level: int) -> Image.Image:
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(img).enhance(level)


def get_augmentations(all_ops: bool = True):
    base = [
        autocontrast,
        equalize,
        posterize,
        rotate,
        solarize,
        shear_x,
        shear_y,
        translate_x,
        translate_y,
    ]
    if not all_ops:
        return base
    return base + [color, contrast, brightness, sharpness]


def clip_image(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)
