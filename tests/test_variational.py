import math

import pytest
import torch

from segloss import analysis as A
from segloss import losses as LS
from segloss.variational import layer as L
from segloss.variational import model as M

D = torch.float64


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _fields(H=12, W=13, aniso=True):
    p = torch.rand(1, 1, H, W, dtype=D)
    w = 0.2 + 0.8 * torch.rand(1, 1, H, W, dtype=D)
    lam = 0.05 + 0.1 * torch.rand(1, 1, H, W, dtype=D)
    if not aniso:
        return p, w, lam, None, None
    a = 2 * math.pi * torch.rand(1, 1, H, W, dtype=D)
    b = 0.1 + 0.9 * torch.rand(1, 1, H, W, dtype=D)
    return p, w, lam, (torch.cos(a), torch.sin(a)), b


def test_adjoint():
    u = torch.randn(2, 1, 9, 11, dtype=D)
    px, py = torch.randn_like(u), torch.randn_like(u)
    gx, gy = L.grad(u)
    px[..., :, -1] = 0
    py[..., -1, :] = 0
    assert torch.allclose((gx * px + gy * py).sum(), -(u * L.div(px, py)).sum())


def test_A_spd_and_bounded():
    p, w, lam, t, b = _fields()
    gx, gy = torch.randn_like(p), torch.randn_like(p)
    ax, ay = L.apply_A(gx, gy, t, b)
    n0 = torch.sqrt(gx ** 2 + gy ** 2)
    n1 = torch.sqrt(ax ** 2 + ay ** 2)
    assert (n1 <= n0 + 1e-12).all() and (n1 >= b * n0 - 1e-12).all()
    assert ((ax * gx + ay * gy) > 0).all()


def test_operator_norm_bound():
    """||grad u||^2 <= 8 ||u||^2 (used for the step size 1 / L = w_min / 8)."""
    for _ in range(20):
        u = torch.randn(1, 1, 15, 17, dtype=D)
        gx, gy = L.grad(u)
        assert float((gx ** 2 + gy ** 2).sum()) <= 8 * float((u ** 2).sum()) + 1e-9


def test_converges_to_minimiser_and_gap():
    for aniso in (False, True):
        p, w, lam, t, b = _fields(aniso=aniso)
        u, (qx, qy) = L.solve(p, w, lam, t, b, K=20000, w_min=0.2, return_dual=True)
        gap = L.duality_gap(u, qx, qy, lam, t, b)
        Pu = L.energy(u, p, w, lam, t, b)
        assert float(gap / Pu) < 1e-6
        assert (u >= 0).all() and (u <= 1).all()
        for _ in range(20):
            v = (u + 1e-3 * torch.randn_like(u)).clamp(0, 1)
            assert float(L.energy(v, p, w, lam, t, b)) >= float(Pu) - 1e-9


def test_gap_decreases_with_K():
    p, w, lam, t, b = _fields()
    gaps = []
    for K in (5, 30, 200, 2000):
        u, (qx, qy) = L.solve(p, w, lam, t, b, K=K, w_min=0.2, return_dual=True)
        gaps.append(float(L.duality_gap(u, qx, qy, lam, t, b) / L.energy(u, p, w, lam, t, b)))
    assert gaps[-1] < gaps[0] and gaps[-1] < 1e-4


def test_lam_zero_is_identity():
    p, w, lam, t, b = _fields()
    u = L.solve(p, w, torch.zeros_like(lam), t, b, K=10, w_min=0.2)
    assert torch.allclose(u, p)


def test_identity_limit_bound_exact_minimiser():
    """w_min ||u* - p||^2 <= J(p) - J(u*) <= J(p) <= sqrt(2) sum(lam)."""
    for aniso in (False, True):
        for scale in (1.0, 0.1, 0.01):
            p, w, lam, t, b = _fields(aniso=aniso)
            lam = lam * scale
            u = L.solve(p, w, lam, t, b, K=20000, w_min=0.2)
            d2 = float(((u - p) ** 2).sum())
            Jp, Ju = float(A.tv_seminorm(p, lam, t, b)), float(A.tv_seminorm(u, lam, t, b))
            assert 0.2 * d2 <= Jp - Ju + 1e-8
            assert Jp <= math.sqrt(2) * float(lam.sum()) + 1e-12
            bd = A.identity_limit_bound(p, lam, 0.2, t, b)
            assert math.sqrt(d2) <= float(bd["exact"]) + 1e-8 <= float(bd["loose"]) + 1e-8


def test_identity_limit_bound_inexact_iterate():
    p, w, lam, t, b = _fields()
    for K in (3, 10, 30):
        u, (qx, qy) = L.solve(p, w, lam, t, b, K=K, w_min=0.2, return_dual=True)
        assert (torch.sqrt(qx ** 2 + qy ** 2) <= lam + 1e-12).all()          # dual iterate is feasible
        gap = L.duality_gap(u, qx, qy, lam, t, b)
        bd = A.identity_limit_bound(p, lam, 0.2, t, b, gap=gap)
        assert float(torch.sqrt(((u - p) ** 2).sum())) <= float(bd["inexact"]) + 1e-10


def test_layer_change_shrinks_with_lambda():
    p, w, lam, t, b = _fields()
    d = [float(((L.solve(p, w, lam * s, t, b, K=2000, w_min=0.2) - p) ** 2).sum()) for s in (1.0, 0.1, 0.01, 0.0)]
    assert d[0] > d[1] > d[2] > d[3] == 0.0


def test_gradcheck_unrolled():
    p, w, lam, t, b = _fields(6, 7)
    p = (0.2 + 0.6 * p).requires_grad_()               # keep away from the box faces (clip kinks)
    w = w.requires_grad_()
    lam = lam.requires_grad_()
    bb = b.clone().requires_grad_()
    f = lambda p, w, lam, bb: L.solve(p, w, lam, t, bb, K=8, w_min=0.2)  # noqa: E731
    assert torch.autograd.gradcheck(f, (p, w, lam, bb), eps=1e-6, atol=1e-5)


def test_gradcheck_orientation():
    p, w, lam, t, b = _fields(6, 7)
    p = 0.2 + 0.6 * p
    o = torch.randn(1, 2, 6, 7, dtype=D, requires_grad=True)

    def f(o):
        n = torch.sqrt(o[:, :1] ** 2 + o[:, 1:] ** 2 + 1e-6)
        return L.solve(p, w, lam, (o[:, :1] / n, o[:, 1:] / n), b, K=8, w_min=0.2)
    assert torch.autograd.gradcheck(f, (o,), eps=1e-6, atol=1e-5)


def test_model_shapes_and_backward():
    for mode in ("none", "tv_iso", "tv_dir", "full"):
        m = M.VariationalSegNet({"mode": mode, "base_filters": 4, "depth": 2, "var": {"K": 5}})
        x = torch.randn(2, 1, 32, 32)
        f = m(x)
        assert f["u"].shape == (2, 1, 32, 32)
        if m.is_var:
            assert (f["u"] >= 0).all() and (f["u"] <= 1).all()
        l, _ = LS.loss("centerline", f, (torch.rand(2, 1, 32, 32) > 0.9).float(), 2, on_logit=not m.is_var)
        l.backward()
        if m.is_var:
            assert m.rho.grad is not None or m.lam_map


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        M.VariationalSegNet({"mode": "refine", "base_filters": 4, "depth": 2})


def test_image_orientation_along_line():
    img = torch.zeros(1, 1, 41, 41)
    img[0, 0, 20, :] = 1
    tx, ty = M.image_orientation(img)
    assert abs(float(tx[0, 0, 20, 20])) > 0.95
