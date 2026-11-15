"""Contrast-weighted tolerance-F1 (CTF) loss for thin-structure segmentation.

    L_CTF = BCE_v(z, y) + (1 - F)

    P = sum_i u_i M_r(y)_i / (sum_i u_i + eps)                      tolerance precision
    R = sum_i w_i y_i M_r(u)_i / (sum_i w_i y_i + eps)              contrast-weighted tolerance recall
    F = 2 P R / (P + R + eps)                                        eps = 1e-6
    BCE_v = mean_i v_i * bce(z_i, y_i),  v_i = 1 + (w_i - 1) y_i     (weights only label terms; mean over all pixels)

* M_r is max pooling over a (2r+1) x (2r+1) window with stride 1 and padding r. PyTorch pads max pooling with -inf,
  so windows are truncated at the border. For a binary label, M_r(y) marks the Chebyshev r-dilation of the label; for
  scores, M_r(u) is the best score within the window. A predicted pixel therefore counts as correct if a label pixel
  lies within r pixels, and a label pixel counts as found if the prediction is high somewhere within r pixels. This
  makes the loss usable with 1-pixel centerline labels without dilating them into masks.
* w = 1 + beta * kappa is the per-pixel contrast weight (kappa = 1 at low image contrast; see `segloss.contrast`).
  It acts on label pixels only, in both the recall term and the BCE term.
* P, R and F are pooled over the whole batch (all crops), not computed per image.
* A batch without label pixels gives P = R = F = 0; the F1 term is then the constant 1 and contributes no gradient.
* In the recall term, max pooling routes the gradient of each label pixel to the arg-max of its window; at ties the
  pixel stored by `max_pool2d` receives it.

`dice_bce` (BCE + soft Dice with smoothing 1) is the reference loss the CTF loss is compared against.

When the scores come from a layer that outputs probabilities directly (``on_logit=False``), BCE is computed on
``clamp(u, 1e-4, 1 - 1e-4)``; otherwise it is computed from the logits (numerically identical in exact arithmetic).
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

EPS = 1e-4

__all__ = ["EPS", "bce", "tol_f1", "soft_dice", "loss", "ctf_loss", "dice_bce_loss", "contrast_weight"]


def contrast_weight(low_contrast: torch.Tensor, beta: float) -> Optional[torch.Tensor]:
    """Per-pixel weight w = 1 + beta * kappa as used in training; ``None`` (unweighted) when beta <= 0."""
    return 1 + beta * low_contrast if beta > 0 else None


def bce(f: Mapping[str, torch.Tensor], y: torch.Tensor, omega: Optional[torch.Tensor] = None,
        on_logit: bool = True) -> torch.Tensor:
    """Binary cross-entropy whose label (y = 1) terms are weighted by ``omega``; mean over all pixels."""
    wt = None if omega is None else 1 + (omega - 1) * y
    if on_logit:
        return F.binary_cross_entropy_with_logits(f["logit"], y, weight=wt)
    return F.binary_cross_entropy(f["u"].clamp(EPS, 1 - EPS), y, weight=wt)


def tol_f1(u: torch.Tensor, y: torch.Tensor, r: int, omega: Optional[torch.Tensor] = None):
    """Soft tolerance F1 with optional label weights. Returns (F, P, R), each a 0-dim tensor.

    u, y, omega: (B, 1, H, W) float tensors; u in [0, 1], y in {0, 1}; r: tolerance radius in pixels."""
    k = 2 * r + 1
    mu = F.max_pool2d(u, k, 1, r)
    my = F.max_pool2d(y, k, 1, r)
    oy = y if omega is None else y * omega
    R = (oy * mu).sum() / (oy.sum() + 1e-6)
    P = (u * my).sum() / (u.sum() + 1e-6)
    return 2 * P * R / (P + R + 1e-6), P, R


def soft_dice(u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - soft Dice with smoothing 1, pooled over the batch."""
    return 1 - (2 * (u * y).sum() + 1) / (u.sum() + y.sum() + 1)


def loss(kind: str, f: Mapping[str, torch.Tensor], y: torch.Tensor, r: int, omega: Optional[torch.Tensor] = None,
         on_logit: bool = True):
    """Training loss by name. ``f`` holds ``"logit"`` and/or ``"u"`` (scores in [0, 1]).

    kind = "centerline": the CTF loss, BCE_omega + (1 - F_tol,omega); returns (loss, {"P": float, "R": float}).
    kind = "dice_bce":   BCE + soft Dice (``omega`` and ``r`` are ignored); returns (loss, {})."""
    if kind == "dice_bce":
        return bce(f, y, None, on_logit) + soft_dice(f["u"], y), {}
    if kind == "centerline":
        F1, P, R = tol_f1(f["u"], y, r, omega)
        return bce(f, y, omega, on_logit) + (1 - F1), {"P": float(P), "R": float(R)}
    raise ValueError(kind)


def ctf_loss(logit: torch.Tensor, y: torch.Tensor, low_contrast: torch.Tensor, r: int, beta: float = 2.0):
    """Convenience wrapper: CTF loss from logits, a binary label and a low-contrast indicator (all (B, 1, H, W))."""
    f = {"logit": logit, "u": torch.sigmoid(logit)}
    return loss("centerline", f, y, r, contrast_weight(low_contrast, beta), on_logit=True)


def dice_bce_loss(logit: torch.Tensor, y: torch.Tensor):
    """Convenience wrapper: BCE + soft Dice from logits."""
    return loss("dice_bce", {"logit": logit, "u": torch.sigmoid(logit)}, y, 0, None, on_logit=True)
