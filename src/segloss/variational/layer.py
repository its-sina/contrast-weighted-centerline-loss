"""Differentiable box-constrained, weighted, anisotropic total-variation (TV) layer, unrolled dual FISTA.

    u*(p, w, lam, A) = argmin_{u in [0,1]^Omega}  1/2 sum_x w(x) (u(x) - p(x))^2  +  sum_x lam(x) |A(x) grad u(x)|_2

A(x) = t t^T + b n n^T  (t a unit vector field, n = t rotated by 90 degrees, b in [b_min, 1]); isotropic TV is A = I.
A(x) is symmetric with eigenvalues 1 and b, so ||A(x)|| <= 1 and A(x) is positive definite: the regulariser is a
convex, 1-homogeneous function of grad u, and with w >= w_min > 0 the problem is w_min-strongly convex (unique
minimiser).

Dual (q(x) in R^2, |q(x)| <= lam(x)):  lam |A g| = max_{|q| <= lam} <q, A g>, so
    D(q) = min_{u in box} 1/2 w (u - p)^2 - <div(A q), u>,   u(q) = clip(p + div(A q) / w, 0, 1)   (closed form)
D is concave with gradient A grad u(q) (Danskin) and Lipschitz constant L <= ||A grad||^2 / w_min <= 8 / w_min.
Projected FISTA on the dual with step 1/L. The layer runs a FIXED number K of iterations and is differentiated
through all of them (unrolled; no implicit differentiation). The output u(q_K) is always feasible (in the box); its
distance to the true minimiser is certified by the exact duality gap
    gap(q) = P(u(q)) - D(q) = sum lam |A grad u(q)| - <q, A grad u(q)>   >= 0,
which bounds P(u_K) - P(u*) and, by strong convexity, ||u_K - u*||^2 <= 2 gap / w_min.
Discretisation: forward differences with a zero last difference (Neumann); div = -grad^T exactly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["grad", "div", "apply_A", "primal_of_dual", "energy", "duality_gap", "solve"]


def grad(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """u (..., H, W) -> forward differences (gx along W, gy along H), zero in the last column / row."""
    gx = F.pad(u[..., :, 1:] - u[..., :, :-1], (0, 1))
    gy = F.pad(u[..., 1:, :] - u[..., :-1, :], (0, 0, 0, 1))
    return gx, gy


def div(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Exact negative adjoint of `grad`: <grad u, p> = -<u, div p>."""
    dx = F.pad(px[..., :, :-1], (0, 1)) - F.pad(px[..., :, :-1], (1, 0))
    dy = F.pad(py[..., :-1, :], (0, 0, 0, 1)) - F.pad(py[..., :-1, :], (0, 0, 1, 0))
    return dx + dy


def apply_A(gx, gy, t, b):
    """A g with A = t t^T + b n n^T; t = (tx, ty) unit, n = (-ty, tx). t None -> identity (isotropic)."""
    if t is None:
        return gx, gy
    tx, ty = t
    a_t = tx * gx + ty * gy           # component along t
    a_n = -ty * gx + tx * gy          # component along n
    a_n = b * a_n
    return tx * a_t - ty * a_n, ty * a_t + tx * a_n


def _proj(qx, qy, lam, eps=1e-12):
    nrm = torch.sqrt(qx * qx + qy * qy + eps)
    s = torch.clamp(lam / nrm, max=1.0)
    return qx * s, qy * s


def primal_of_dual(p, w, qx, qy, t, b):
    ax, ay = apply_A(qx, qy, t, b)                       # A symmetric: A^T q = A q
    return torch.clamp(p + div(ax, ay) / w, 0.0, 1.0)


def energy(u, p, w, lam, t, b):
    gx, gy = grad(u)
    ax, ay = apply_A(gx, gy, t, b)
    return (0.5 * w * (u - p) ** 2).sum((-2, -1)) + (lam * torch.sqrt(ax * ax + ay * ay + 1e-30)).sum((-2, -1))


def duality_gap(u, qx, qy, lam, t, b):
    gx, gy = grad(u)
    ax, ay = apply_A(gx, gy, t, b)
    return (lam * torch.sqrt(ax * ax + ay * ay + 1e-30) - (qx * ax + qy * ay)).sum((-2, -1))


def solve(p: torch.Tensor, w: torch.Tensor | float, lam: torch.Tensor | float, t=None, b=None, K: int = 30,
          w_min: float | None = None, return_dual: bool = False):
    """Unrolled dual FISTA, K iterations. p (..., H, W) in [0, 1]; w >= w_min > 0; lam >= 0 (scalar or map);
    t = (tx, ty) unit fields or None (isotropic); b in (0, 1] map or None. Differentiable w.r.t. all inputs."""
    if not torch.is_tensor(w):
        w = torch.full_like(p, float(w))
    if not torch.is_tensor(lam):
        lam = torch.full_like(p, float(lam))
    if b is None and t is not None:
        b = torch.ones_like(p)
    wm = float(w.min().detach()) if w_min is None else w_min
    tau = wm / 8.0                                       # 1 / L,  L = ||A grad||^2 / w_min <= 8 / w_min
    qx = torch.zeros_like(p)
    qy = torch.zeros_like(p)
    zx, zy, th = qx, qy, 1.0
    for _ in range(K):
        u = primal_of_dual(p, w, zx, zy, t, b)
        gx, gy = grad(u)
        ax, ay = apply_A(gx, gy, t, b)
        nx, ny = _proj(zx + tau * ax, zy + tau * ay, lam)
        th_n = 0.5 * (1 + (1 + 4 * th * th) ** 0.5)
        mom = (th - 1) / th_n
        zx, zy = nx + mom * (nx - qx), ny + mom * (ny - qy)
        qx, qy, th = nx, ny, th_n
    u = primal_of_dual(p, w, qx, qy, t, b)
    if return_dual:
        return u, (qx, qy)
    return u
