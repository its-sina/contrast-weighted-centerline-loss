"""Operating point at a fixed false-positive burden.

Lowering the threshold trivially raises recall, so methods are compared at thresholds chosen to give the same mean
false-positive burden on a set of fitting images. ``match`` bisects on [0, 1] for the threshold at which the mean
burden of the binarised score maps reaches ``target`` and returns the upper end of the final bracket, which
satisfies mean FP <= target. Skeletonisation makes the burden not necessarily monotone in the threshold, so this is
a feasible threshold at the bisection boundary, not a guaranteed global minimum. If even the highest threshold
exceeds the target, the threshold is returned with ``matched=False``. The fitted thresholds are then frozen and
applied unchanged to new images, whose realised burden is reported rather than matched.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from segloss.metrics import CenterlineReference

__all__ = ["fp_at", "match"]


def fp_at(maps: Mapping[str, np.ndarray], R: Mapping[str, CenterlineReference], th: float) -> float:
    """Mean false-positive burden of {score_map > th} over the images in ``maps``."""
    return float(np.mean([R[c].evaluate(maps[c] > th)["fp_burden"] for c in maps]))


def match(maps: Dict[str, np.ndarray], R: Dict[str, CenterlineReference], target: float,
          iters: int = 18) -> Tuple[float, bool]:
    """(threshold, matched): bisection for mean FP burden <= target."""
    lo, hi = 0.0, 1.0
    if fp_at(maps, R, hi - 1e-6) > target:
        return hi - 1e-6, False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fp_at(maps, R, mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi, True
