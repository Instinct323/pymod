import numpy as np


def remove_background(img: np.ndarray,
                      color_bg: int | tuple,
                      thresh: int = 10) -> np.ndarray:
    """ Remove background from image. """
    mask = np.abs(img.astype(int) - color_bg).sum(axis=-1) >= thresh
    alpha = mask.astype(np.uint8) * 255
    return np.concatenate([img, alpha[..., None]], axis=-1)
