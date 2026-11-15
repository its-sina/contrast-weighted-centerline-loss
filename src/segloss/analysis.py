"""Closed-form quantities for the CTF loss and the TV layer, used to test and explain their behaviour.

ctf_logit_gradient       exact gradient of loss("centerline") with respect to the logits, written out by hand:
                           dP/du_k = (M_r(y)_k - P) / (S_u + eps)
                           dR/du_k = (1 / (S_w + eps)) * sum_{i: pi(i) = k} w_i y_i      (pi(i) = arg-max of i's window)
                           dF/dP = 2R(R + eps) / d^2,  dF/dR = 2P(P + eps) / d^2,  d = P + R + eps
                           dL/dz_k = v_k (u_k - y_k) / N - u_k (1 - u_k) (dF/dP dP/du_k + dF/dR dR/du_k)
                         The recall gradient of each label pixel is routed to the arg-max of its window; a pixel that
                         is the arg-max for several label pixels accumulates their weights.
recall_weights           per-pixel recall weights of low- and high-contrast label pixels and the share of recall
                         carried by low-contrast pixels: the contrast weight reallocates recall, it does not scale it.
tv_seminorm              J_lam(u) = sum lam |A grad u|.
identity_limit_bound     for the exact TV minimiser u* with p in [0,1], w >= w_min > 0, ||A|| <= 1:
                           w_min ||u* - p||^2 <= J_lam(p) - J_lam(u*) <= J_lam(p) <= sqrt(2) sum lam
                         and for an iterate u_K = u(q_K) with feasible dual q_K and duality gap G:
                           ||u_K - p|| <= sqrt(J_lam(p) / w_min) + sqrt(2 G / w_min).
surrogate_upper_bound    for a binary output S and r = tol, every label pixel within Euclidean distance tol of the
                         skeleton of S lies in the r-window of S, so evaluation recall <= training recall at u = 1_S.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from segloss.metrics import skel
from segloss.variational import layer as L

__all__ = ["ctf_logit_gradient", "recall_weights", "tv_seminorm", "identity_limit_bound", "surrogate_upper_bound"]


@torch.no_grad()
def ctf_logit_gradient(z: torch.Tensor, y: torch.Tensor, r: int, omega: Optional[torch.Tensor] = None,
                       eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Hand-derived gradient of ``losses.loss("centerline", ...)`` w.r.t. the logits z (all tensors (B, 1, H, W)).

    Valid where every window maximum that carries label weight is unique; at ties the pixel stored by
    ``max_pool2d`` receives the weight, as in autograd. Returns grad, the BCE and F1 parts, P, R and ``routed``
    (number of label pixels whose window arg-max is each pixel)."""
    u = torch.sigmoid(z)
    w = torch.ones_like(u) if omega is None else omega
    k = 2 * r + 1
    my = F.max_pool2d(y, k, 1, r)
    mu, idx = F.max_pool2d(u, k, 1, r, return_indices=True)
    Su, Sw = u.sum(), (w * y).sum()
    P = (u * my).sum() / (Su + eps)
    R = (w * y * mu).sum() / (Sw + eps)
    d = P + R + eps
    dP = (my - P) / (Su + eps)
    flat = (u.shape[0], u.shape[1], -1)
    dR = torch.zeros_like(u).reshape(flat)
    dR.scatter_add_(2, idx.reshape(flat), (w * y).reshape(flat) / (Sw + eps))
    dR = dR.view_as(u)
    routed = torch.zeros_like(u).reshape(flat)
    routed.scatter_add_(2, idx.reshape(flat), (y > 0).to(u.dtype).reshape(flat))
    FP, FR = 2 * R * (R + eps) / d ** 2, 2 * P * (P + eps) / d ** 2
    v = 1 + (w - 1) * y
    g_bce = v * (u - y) / u.numel()
    g_f1 = -u * (1 - u) * (FP * dP + FR * dR)
    return {"grad": g_bce + g_f1, "grad_bce": g_bce, "grad_f1": g_f1, "P": P, "R": R,
            "routed": routed.view_as(u)}


def recall_weights(n_low: int, n_high: int, beta: float) -> Dict[str, float]:
    """Recall weight per low- and per high-contrast label pixel, and the low-contrast share of R (eps ignored)."""
    S = n_high + (1 + beta) * n_low
    return {"per_low": (1 + beta) / S, "per_high": 1 / S, "low_share": (1 + beta) * n_low / S}


def tv_seminorm(u: torch.Tensor, lam: torch.Tensor, t=None, b=None) -> torch.Tensor:
    """J_lam(u) = sum_x lam(x) |A(x) grad u(x)|_2, summed over the last two dimensions."""
    gx, gy = L.grad(u)
    ax, ay = L.apply_A(gx, gy, t, b)
    return (lam * torch.sqrt(ax * ax + ay * ay + 1e-30)).sum((-2, -1))


def identity_limit_bound(p: torch.Tensor, lam: torch.Tensor, w_min: float, t=None, b=None,
                         gap: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Upper bounds on ||u - p||_2 per image: exact minimiser (``exact``), its loose form from sum(lam)
    (``loose``), and, if the duality gap of an iterate is given, the bound for that iterate (``inexact``)."""
    J = tv_seminorm(p, lam, t, b)
    out = {"J_p": J, "exact": torch.sqrt(J / w_min),
           "loose": torch.sqrt(math.sqrt(2) * lam.sum((-2, -1)) / w_min)}
    if gap is not None:
        out["inexact"] = out["exact"] + torch.sqrt(2 * gap.clamp_min(0) / w_min)
    return out


def surrogate_upper_bound(S: np.ndarray, label: np.ndarray, low_contrast: np.ndarray, tol: int) -> Dict[str, float]:
    """Evaluation low-contrast recall of binary S (Euclidean, skeleton) and the training-side recall at u = 1_S
    (square window of radius tol on the binary map), over the low-contrast label-skeleton pixels."""
    Ls = skel(label) & low_contrast.astype(bool)
    if not Ls.any():
        return {"eval_recall": float("nan"), "train_recall": float("nan")}
    Sp = skel(S)
    dp = ndi.distance_transform_edt(~Sp) if Sp.any() else np.full(S.shape, np.inf)
    win = ndi.maximum_filter(S.astype(np.uint8), size=2 * tol + 1, mode="constant") > 0
    return {"eval_recall": float((dp[Ls] <= tol).mean()), "train_recall": float(win[Ls].mean())}
