"""Crop quality metrics — reject pixelated / blurry crops (numpy only, no cv2)."""

from __future__ import annotations

from PIL import Image


def laplacian_var(img: Image.Image) -> float:
    """Variance of the Laplacian = sharpness. Low value => blurry / pixelated.

    Computed on the NATIVE crop (before any upscale) so an upscaled tiny detection
    still reads as blurry.
    """
    import numpy as np
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4.0 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    return float(lap[1:-1, 1:-1].var())
