"""U-Net backbone with an optional end-to-end trainable TV layer (`segloss.variational.layer`).

Output channels of the shared U-Net decoder (1x1 conv), by mode:
  none:            logit                        u = p = sigmoid(logit)
  tv_iso:          logit                        scalar lam, w = 1, A = I
  tv_dir / full:   o1, o2                       learned orientation t = (o1, o2) / |(o1, o2)|  (only t t^T enters)
                   o3                           across-vessel weight b = b_min + (1 - b_min) sigmoid(o3) in [b_min, 1]
  full:            o4                           fidelity weight w = w_min + (1 - w_min) sigmoid(o4) in [w_min, 1]
                   o5                           spatial regularisation lam(x) = lam_max sigmoid(o5)
Scalar lam (tv_iso / tv_dir) = lam_max sigmoid(rho) with a trainable rho (frozen if var["train_lam"] is False).
Orientation "image": t from the structure tensor of the input image (fixed, not learned).
For the variational modes u is the layer output after K unrolled iterations and p the backbone probability.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from segloss.models.unet import UNet
from segloss.variational import layer as L

__all__ = ["image_orientation", "VariationalSegNet", "n_params"]


def _gauss1d(sigma: float, device, dtype):
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def _smooth(x, sigma):
    g = _gauss1d(sigma, x.device, x.dtype)
    r = (len(g) - 1) // 2
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def image_orientation(img: torch.Tensor, sigma_d: float = 1.0, sigma_i: float = 3.0):
    """Unit field along the local structure direction = eigenvector of the smallest structure-tensor eigenvalue."""
    s = _smooth(img, sigma_d)
    gx, gy = L.grad(s)
    J = _smooth(torch.cat([gx * gx, gx * gy, gy * gy], 1), sigma_i)
    jxx, jxy, jyy = J[:, :1], J[:, 1:2], J[:, 2:3]
    ang = 0.5 * torch.atan2(2 * jxy, jxx - jyy)        # dominant gradient direction
    return -torch.sin(ang), torch.cos(ang)             # perpendicular = along the structure


_COMPILED: Dict[str, object] = {}


def compiled_solve():
    """torch.compile'd unrolled solver (same arithmetic as the eager one)."""
    if "f" not in _COMPILED:
        _COMPILED["f"] = torch.compile(L.solve, dynamic=False)
    return _COMPILED["f"]


class VariationalSegNet(nn.Module):
    """cfg: base_filters, depth, mode in {none, tv_iso, tv_dir, full}, optional var = {K, lam_max, w_min, b_min,
    orient, learn_w, lam_map, learn_b, lam_init, train_lam}, optional compile (default True; CUDA only)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.mode = cfg.get("mode", "none")
        if self.mode not in ("none", "tv_iso", "tv_dir", "full"):
            raise ValueError(self.mode)
        v = cfg.get("var", {})
        self.K = int(v.get("K", 30))
        self.lam_max = float(v.get("lam_max", 0.2))
        self.w_min = float(v.get("w_min", 0.2))
        self.b_min = float(v.get("b_min", 0.1))
        self.orient = v.get("orient", "learned")         # learned | image
        self.learn_w = bool(v.get("learn_w", self.mode == "full"))
        self.lam_map = bool(v.get("lam_map", self.mode == "full"))
        self.learn_b = bool(v.get("learn_b", True))
        self.use_compile = bool(cfg.get("compile", True))
        n_out = 1
        if self.mode in ("tv_dir", "full"):
            n_out = 4
        if self.mode == "full":
            n_out = 6
        self.net = UNet(in_channels=1, out_channels=n_out, base_filters=cfg["base_filters"], depth=cfg["depth"])
        lam0 = float(v.get("lam_init", 0.05))
        rho = torch.tensor(math.log(lam0 / (self.lam_max - lam0)))
        self.rho = nn.Parameter(rho, requires_grad=bool(v.get("train_lam", True)))

    @property
    def wmin_eff(self) -> float:
        """Lower bound of w used for the step size (w = 1 when the fidelity weight is not learned)."""
        return self.w_min if self.learn_w else 1.0

    @property
    def is_var(self) -> bool:
        return self.mode in ("tv_iso", "tv_dir", "full")

    def fields(self, x, o):
        """Variational parameters from the network output o (float32) and the image x."""
        p = torch.sigmoid(o[:, :1])
        f = {"p": p, "logit": o[:, :1]}
        if not self.is_var:
            return f
        if self.lam_map:
            f["lam"] = self.lam_max * torch.sigmoid(o[:, 5:6])
        else:
            f["lam"] = (self.lam_max * torch.sigmoid(self.rho)).expand_as(p)
        f["w"] = self.w_min + (1 - self.w_min) * torch.sigmoid(o[:, 4:5]) if self.learn_w else torch.ones_like(p)
        if self.mode == "tv_iso":
            f["t"], f["b"] = None, None
            return f
        if self.orient == "image":
            tx, ty = image_orientation(x.float())
        else:
            n = torch.sqrt(o[:, 1:2] ** 2 + o[:, 2:3] ** 2 + 1e-6)
            tx, ty = o[:, 1:2] / n, o[:, 2:3] / n
        f["t"] = (tx, ty)
        f["b"] = (self.b_min + (1 - self.b_min) * torch.sigmoid(o[:, 3:4])) if self.learn_b else \
            torch.full_like(p, self.b_min)
        return f

    def forward(self, x, K: Optional[int] = None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            o = self.net(x)
        o = o.float().contiguous()
        f = self.fields(x, o)
        if self.is_var:
            solver = compiled_solve() if (self.use_compile and x.is_cuda and K is None) else L.solve
            f["u"] = solver(f["p"], f["w"], f["lam"], f["t"], f["b"], K=self.K if K is None else K, w_min=self.wmin_eff)
        else:
            f["u"] = f["p"]
        return f


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
