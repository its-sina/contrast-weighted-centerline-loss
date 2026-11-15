"""Minimal training integration of the CTF loss: crop sampler, model forward, optimiser / schedule, one step.

This is the training loop the loss was used in, reduced to the parts that touch the loss:
  * images are standardised per image; the low-contrast indicator kappa is computed once per full image from the
    raw intensities and is cropped / flipped / rotated together with image and label (the random intensity
    scale-and-shift is applied to the network input only);
  * the network output is a single logit map; scores u = sigmoid(logit); on CUDA the network runs under bf16
    autocast and the loss is computed in float32;
  * omega = 1 + beta * kappa for the CTF loss (None when beta = 0), then ``segloss.losses.loss``;
  * AdamW, linear warm-up then cosine decay (per step), global gradient-norm clipping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from segloss import losses as LS

__all__ = ["TrainConfig", "normalize", "Crops", "forward_scores", "make_optimizer", "train_step"]


@dataclass
class TrainConfig:
    loss: str = "centerline"          # "centerline" (CTF) or "dice_bce"
    beta: float = 2.0                 # contrast weight; 0 disables it in both terms
    radius: int = 2                   # tolerance radius r in pixels
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 200
    iters: int = 8000
    clip: float = 5.0                 # global L2 gradient-norm clip


def normalize(img: np.ndarray) -> np.ndarray:
    """Per-image standardisation; returns (1, H, W) float32."""
    x = img.astype(np.float32)
    return ((x - x.mean()) / (x.std() + 1e-6))[None]


class Crops(Dataset):
    """Random crops with flip / rot90 / intensity augmentation, carrying the low-contrast indicator along.

    imgs: list of (1, H, W) normalised images; labs: list of (H, W) binary labels; lowc: list of (H, W) indicators.
    Each item is (x (1, s, s), y (1, s, s), kappa (1, s, s)); item k is deterministic given (seed, k)."""

    def __init__(self, imgs: Sequence[np.ndarray], labs: Sequence[np.ndarray], lowc: Sequence[np.ndarray],
                 crop: int, length: int, seed: Tuple[int, ...]):
        self.imgs, self.labs, self.lowc, self.crop, self.length, self.seed = imgs, labs, lowc, crop, length, seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, k: int):
        rng = np.random.default_rng((*self.seed, k))
        i = int(rng.integers(len(self.imgs)))
        x, m, c = self.imgs[i], self.labs[i], self.lowc[i]
        H, W = m.shape
        s = self.crop
        y0, x0 = int(rng.integers(0, H - s + 1)), int(rng.integers(0, W - s + 1))
        x, m, c = x[:, y0:y0 + s, x0:x0 + s], m[y0:y0 + s, x0:x0 + s].astype(np.float32), c[y0:y0 + s, x0:x0 + s]
        x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.2, 0.2)
        if rng.random() < 0.5:
            x, m, c = x[:, :, ::-1], m[:, ::-1], c[:, ::-1]
        if rng.random() < 0.5:
            x, m, c = x[:, ::-1], m[::-1], c[::-1]
        r = int(rng.integers(4))
        x, m, c = np.rot90(x, r, (1, 2)), np.rot90(m, r), np.rot90(c, r)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))  # noqa: E731
        return f(x), f(m)[None], f(c)[None]


def forward_scores(net: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Logit, probability p and score u (= p) of a single-logit network; bf16 autocast on CUDA only."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        o = net(x)
    o = o.float().contiguous()
    p = torch.sigmoid(o[:, :1])
    return {"p": p, "logit": o[:, :1], "u": p}


def make_optimizer(net: torch.nn.Module, cfg: TrainConfig):
    """AdamW with a per-step schedule: linear warm-up over ``warmup`` steps times cosine decay over ``iters``."""
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total, warm = cfg.iters, cfg.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    return opt, sched


def train_step(net: torch.nn.Module, opt: torch.optim.Optimizer, sched, batch: List[torch.Tensor],
               cfg: TrainConfig) -> Dict[str, float]:
    """One optimisation step on a batch (x, y, kappa); returns the loss, gradient norm and loss diagnostics."""
    x, y, c = batch
    f = forward_scores(net, x)
    omega = 1 + cfg.beta * c if cfg.loss == "centerline" and cfg.beta > 0 else None
    loss, info = LS.loss(cfg.loss, f, y, cfg.radius, omega, on_logit=True)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
    opt.step()
    sched.step()
    return {"loss": float(loss), "grad_norm": float(gn), **info}
