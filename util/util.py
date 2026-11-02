import numpy as np
from processing import spatial

def to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def to_gray(img):
    return spatial.rgb_to_gray(img) if img.ndim == 3 else img


def as_rgb(img):
    return img if img.ndim == 3 else np.stack([img] * 3, axis=-1)