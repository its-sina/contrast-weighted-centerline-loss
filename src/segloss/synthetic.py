"""Synthetic thin bright curves on a noisy background, with 1-pixel centerline labels.

Each image contains ``n_curves`` smooth random curves. Every curve gets its own peak intensity, so the set ranges
from clearly visible to barely above the noise, which gives the low-contrast indicator something to separate.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

__all__ = ["random_curve", "synthetic_image", "synthetic_set"]


def random_curve(rng: np.random.Generator, size: int, steps: int = 400, step_len: float = 1.0,
                 turn_sd: float = 0.08) -> np.ndarray:
    """Points of a smooth random walk with persistent heading, clipped to the image; returns (n, 2) int (row, col)."""
    pos = rng.uniform(0.15, 0.85, 2) * size
    ang = rng.uniform(0, 2 * np.pi)
    pts = []
    for _ in range(steps):
        ang += rng.normal(0, turn_sd)
        pos = pos + step_len * np.array([np.sin(ang), np.cos(ang)])
        if not (0 <= pos[0] < size and 0 <= pos[1] < size):
            break
        pts.append(np.floor(pos).astype(int))
    return np.unique(np.array(pts, int).reshape(-1, 2), axis=0) if pts else np.zeros((0, 2), int)


def synthetic_image(rng: np.random.Generator, size: int = 128, n_curves: int = 8, noise: float = 0.25,
                    width: float = 0.8, intensity: Tuple[float, float] = (0.15, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """(image float32 (size, size), label bool (size, size) with 1-pixel centerlines)."""
    label = np.zeros((size, size), bool)
    signal = np.zeros((size, size), np.float64)
    for _ in range(n_curves):
        p = random_curve(rng, size)
        if len(p) == 0:
            continue
        line = np.zeros((size, size), np.float64)
        line[p[:, 0], p[:, 1]] = 1.0
        label[p[:, 0], p[:, 1]] = True
        signal = np.maximum(signal, rng.uniform(*intensity) * ndi.gaussian_filter(line, width) / 0.2)
    background = ndi.gaussian_filter(rng.normal(0, 1, (size, size)), 6) * 0.3
    img = signal + background + rng.normal(0, noise, (size, size))
    return img.astype(np.float32), label


def synthetic_set(n: int, size: int = 128, seed: int = 0, **kw) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """n synthetic (image, label) pairs from one seed."""
    rng = np.random.default_rng(seed)
    imgs, labs = [], []
    for _ in range(n):
        i, l = synthetic_image(rng, size, **kw)
        imgs.append(i)
        labs.append(l)
    return imgs, labs
