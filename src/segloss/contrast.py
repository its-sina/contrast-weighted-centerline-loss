"""Image-only local contrast and the low-contrast indicator used to weight the CTF loss.

    c = (G_sigma * I - B) / S,   B = median of I over a disc of radius r_bg,
                                 S = 1.4826 * median of |I - B| over the same disc + 1e-3

for images in which the structures of interest are bright on a darker background. B and S are robust local estimates
of background level and noise, so c is a local signal-to-noise ratio. Medians use reflecting boundaries.

The threshold c_thr is the 1/3 quantile of c over label-skeleton pixels of a set of fitting images, and the training
indicator is kappa = 1[c < c_thr], evaluated at every pixel (the loss applies it to label pixels only).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage as ndi

__all__ = ["contrast_map", "low_contrast_indicator", "fit_low_contrast_threshold", "skeleton"]


def contrast_map(img: np.ndarray, r_bg: int = 7, sigma: float = 0.7) -> np.ndarray:
    """(G_sigma * I - B) / S with B = local median (disc radius r_bg), S = 1.4826 x local median of |I - B|."""
    x = img.astype(np.float64)
    yy, xx = np.mgrid[-r_bg:r_bg + 1, -r_bg:r_bg + 1]
    fp = (yy ** 2 + xx ** 2) <= r_bg ** 2
    B = ndi.median_filter(x, footprint=fp, mode="reflect")
    S = 1.4826 * ndi.median_filter(np.abs(x - B), footprint=fp, mode="reflect") + 1e-3
    return (ndi.gaussian_filter(x, sigma) - B) / S


def low_contrast_indicator(img: np.ndarray, c_thr: float) -> np.ndarray:
    """kappa = 1[contrast_map(img) < c_thr] as float32, at every pixel."""
    return (contrast_map(img) < c_thr).astype(np.float32)


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton of a binary mask (empty mask -> all False)."""
    from skimage.morphology import skeletonize
    return skeletonize(mask.astype(bool)) if mask.any() else np.zeros_like(mask, bool)


def fit_low_contrast_threshold(images: Iterable[np.ndarray], labels: Iterable[np.ndarray],
                               q: float = 1 / 3) -> float:
    """c_thr = q-quantile of the contrast at label-skeleton pixels, pooled over all fitting images."""
    vals = [contrast_map(img)[skeleton(lab)] for img, lab in zip(images, labels)]
    return float(np.quantile(np.concatenate(vals), q))
