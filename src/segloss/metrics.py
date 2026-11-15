"""Tolerance-based centerline evaluation of a binary output against a label.

For a binary output S with skeleton Sp, the label skeleton Sl (the label itself for 1-pixel centerline labels;
skeletonised for full-width labels), Euclidean distance transforms and a tolerance tol (pixels):
  cl_recall / cl_precision / cl_f1   recall of Sl pixels within tol of Sp; precision of Sp pixels within tol of Sl
  lc_recall / hc_recall              recall over Sl pixels whose image-only contrast is below / at-or-above c_thr
  fp_burden                          off-label predicted centerline length per area (mm / mm^2):
                                     #(Sp pixels farther than tol from Sl) x pixel size / field area
  len_density, len_density_label     skeleton length density (mm / mm^2) of S and of the label; density_bias = relative
  junction_recall                    recall over Sl pixels within 4 px of a label junction (skeleton degree >= 3)
  dice, cldice                       only for full-width labels (``full_width=True``)
Training uses a square max-pool window on label pixels; this evaluation thresholds, skeletonises and uses Euclidean
distances, so the two neighbourhoods differ at equal radius (see `segloss.analysis.surrogate_upper_bound`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

__all__ = ["skel", "junctions", "CenterlineReference", "merged_components", "stratified_recall"]

S8 = np.ones((3, 3), bool)


def skel(m: np.ndarray) -> np.ndarray:
    return skeletonize(m.astype(bool)) if m.any() else np.zeros_like(m, bool)


def junctions(sk: np.ndarray) -> np.ndarray:
    n = ndi.convolve(sk.astype(np.uint8), S8.astype(np.uint8), mode="constant") - sk
    return sk & (n >= 3)


class CenterlineReference:
    """Precomputed label-side quantities for one image (built once, reused for every method / threshold).

    label: (H, W) binary; contrast: (H, W) image-only contrast; c_thr: low-contrast threshold; tol: tolerance (px);
    um_per_px: pixel size in micrometres; full_width: True if the label has full-width (mask) components."""

    def __init__(self, label: np.ndarray, contrast: np.ndarray, c_thr: float, tol: float, um_per_px: float,
                 full_width: bool):
        self.label = label.astype(bool)
        self.Sl = skel(self.label)
        self.dl = ndi.distance_transform_edt(~self.Sl)
        self.tol, self.um, self.mixed = tol, um_per_px, full_width
        self.lc = self.Sl & (contrast < c_thr)
        self.hc = self.Sl & (contrast >= c_thr)
        J = junctions(self.Sl)
        self.near_j = self.Sl & (ndi.distance_transform_edt(~J) <= 4) if J.any() else np.zeros_like(self.Sl)
        self.area_mm2 = self.label.size * (um_per_px / 1000) ** 2
        self.len_label = self.Sl.sum() * um_per_px / 1000 / self.area_mm2

    def evaluate(self, S: np.ndarray, Sp: Optional[np.ndarray] = None) -> dict:
        S = S.astype(bool)
        Sp = skel(S) if Sp is None else Sp
        dp = ndi.distance_transform_edt(~Sp) if Sp.any() else np.full(S.shape, np.inf)
        cov = dp <= self.tol
        on = self.dl[Sp] <= self.tol
        prec = float(on.mean()) if Sp.any() else 0.0
        rec = float(cov[self.Sl].mean()) if self.Sl.any() else float("nan")
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        k = self.um / 1000 / self.area_mm2
        out = {"cl_precision": prec, "cl_recall": rec, "cl_f1": f1,
               "lc_recall": float(cov[self.lc].mean()) if self.lc.any() else float("nan"),
               "hc_recall": float(cov[self.hc].mean()) if self.hc.any() else float("nan"),
               "n_lc": int(self.lc.sum()), "n_hc": int(self.hc.sum()),
               "junction_recall": float(cov[self.near_j].mean()) if self.near_j.any() else float("nan"),
               "fp_burden": float((~on).sum() * k), "len_density": float(Sp.sum() * k),
               "len_density_label": float(self.len_label),
               "fg_fraction": float(S.mean())}
        out["density_bias"] = (out["len_density"] - self.len_label) / max(self.len_label, 1e-12)
        if self.mixed:
            out["dice"] = float(2 * (S & self.label).sum() / max(S.sum() + self.label.sum(), 1))
            tp = (Sp & self.label).sum() / max(Sp.sum(), 1)
            ts = (self.Sl & S).sum() / max(self.Sl.sum(), 1)
            out["cldice"] = float(2 * tp * ts / max(tp + ts, 1e-12))
        return out


def merged_components(raw: np.ndarray, out: np.ndarray) -> int:
    """Number of distinct 8-components of ``raw`` that became connected to another raw component in ``out | raw``."""
    lr, nr = ndi.label(raw, S8)
    lo, _ = ndi.label(out | raw, S8)
    if nr == 0:
        return 0
    host = ndi.maximum(lo, lr, range(1, nr + 1))
    _, counts = np.unique(np.asarray(host), return_counts=True)
    return int(counts[counts > 1].sum())


def stratified_recall(ref: CenterlineReference, contrast: np.ndarray, edges: np.ndarray, S: np.ndarray) -> dict:
    """Recall of label-skeleton pixels within tolerance, per image-contrast bin (bins between ``edges``)."""
    Sp = skel(S)
    dp = ndi.distance_transform_edt(~Sp) if Sp.any() else np.full(S.shape, np.inf)
    cov = dp <= ref.tol
    b = np.digitize(contrast[ref.Sl], edges)
    c = cov[ref.Sl]
    return {f"strat{k}": (float(c[b == k].mean()) if (b == k).any() else float("nan")) for k in range(len(edges) + 1)}
