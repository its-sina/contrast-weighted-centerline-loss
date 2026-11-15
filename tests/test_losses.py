import pytest
import torch

from segloss import losses as LS


def line_batch(n=40, rows=7, label_row=3, pred_row=3, eta=1e-3):
    """One crop: a horizontal 1-px label line, and scores 1 - eta on pred_row, eta elsewhere."""
    y = torch.zeros(1, 1, rows, n)
    y[..., label_row, :] = 1
    u = torch.full((1, 1, rows, n), eta)
    u[..., pred_row, :] = 1 - eta
    return u, y


def test_offset_is_free_with_tolerance_but_not_without():
    u, y = line_batch(pred_row=4)
    F0, _, _ = LS.tol_f1(u, y, r=0)
    F1, P1, R1 = LS.tol_f1(u, y, r=1)
    assert float(F0) < 0.01
    assert float(F1) > 0.99 and float(P1) > 0.99 and float(R1) > 0.99


def test_bce_keeps_offset_penalty():
    u, y = line_batch(pred_row=4)
    f = {"u": u, "logit": torch.logit(u)}
    shifted = LS.bce(f, y)
    u0, _ = line_batch(pred_row=3)
    aligned = LS.bce({"u": u0, "logit": torch.logit(u0)}, y)
    assert float(shifted) > 10 * float(aligned)


def test_empty_label_batch_gives_no_f1_gradient():
    z = torch.randn(2, 1, 16, 16, requires_grad=True)
    y = torch.zeros(2, 1, 16, 16)
    F1, P, R = LS.tol_f1(torch.sigmoid(z), y, r=2)
    assert float(F1) == 0 and float(P) == 0 and float(R) == 0
    (1 - F1).backward()
    assert torch.count_nonzero(z.grad) == 0


def test_recall_gradient_is_routed_to_window_maximum():
    y = torch.zeros(1, 1, 9, 9)
    y[0, 0, 4, 4] = 1
    u = torch.full((1, 1, 9, 9), 0.1)
    u[0, 0, 5, 6] = 0.7                       # unique maximum inside the r = 2 window of the label pixel
    u.requires_grad_(True)
    _, _, R = LS.tol_f1(u, y, r=2)
    R.backward()
    g = u.grad[0, 0]
    assert torch.argmax(g).item() == 5 * 9 + 6
    assert torch.count_nonzero(g) == 1
    assert torch.allclose(g[5, 6], torch.tensor(1 / (1 + 1e-6)))


def test_bce_weight_applies_to_label_terms_and_is_mean_over_pixels():
    z = torch.randn(1, 1, 8, 8)
    y = (torch.rand(1, 1, 8, 8) > 0.7).float()
    kappa = (torch.rand(1, 1, 8, 8) > 0.5).float()
    omega = 1 + 2.0 * kappa
    got = LS.bce({"logit": z}, y, omega)
    per = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
    want = ((1 + 2.0 * kappa * y) * per).sum() / z.numel()
    assert torch.allclose(got, want, atol=1e-6)


def test_beta_zero_is_unweighted():
    assert LS.contrast_weight(torch.ones(1), 0.0) is None
    z = torch.randn(2, 1, 12, 12)
    y = (torch.rand(2, 1, 12, 12) > 0.8).float()
    k = (torch.rand(2, 1, 12, 12) > 0.5).float()
    a, _ = LS.ctf_loss(z, y, k, r=1, beta=0.0)
    b, _ = LS.loss("centerline", {"logit": z, "u": torch.sigmoid(z)}, y, 1, None)
    assert torch.equal(a, b)


def test_contrast_weight_reallocates_recall():
    """With one low- and one high-contrast label pixel, beta raises the weight of the former relative to the latter."""
    y = torch.zeros(1, 1, 1, 9)
    y[..., 1] = 1
    y[..., 7] = 1
    k = torch.zeros_like(y)
    k[..., 1] = 1
    u = torch.full_like(y, 0.2).requires_grad_(True)
    _, _, R = LS.tol_f1(u, y, r=0, omega=1 + 2.0 * k)
    R.backward()
    assert torch.allclose(u.grad[0, 0, 0, 1] / u.grad[0, 0, 0, 7], torch.tensor(3.0))


def test_logit_gradient_matches_closed_form():
    torch.manual_seed(0)
    z = torch.randn(2, 1, 10, 10, dtype=torch.float64, requires_grad=True)
    y = (torch.rand(2, 1, 10, 10) > 0.75).double()
    k = (torch.rand(2, 1, 10, 10) > 0.5).double()
    r, beta, eps = 1, 2.0, 1e-6
    omega = 1 + beta * k
    L, _ = LS.loss("centerline", {"logit": z, "u": torch.sigmoid(z)}, y, r, omega)
    L.backward()
    u = torch.sigmoid(z.detach())
    N = u.numel()
    mp = torch.nn.functional.max_pool2d
    my = mp(y, 2 * r + 1, 1, r)
    mu, idx = mp(u, 2 * r + 1, 1, r, return_indices=True)
    Su, So = u.sum(), (omega * y).sum()
    P = (u * my).sum() / (Su + eps)
    R = (omega * y * mu).sum() / (So + eps)
    d = P + R + eps
    dP = (my - P) / (Su + eps)
    dR = torch.zeros_like(u).flatten(2)
    dR.scatter_add_(2, idx.flatten(2), (omega * y).flatten(2) / (So + eps))
    dR = dR.view_as(u)
    FP, FR = 2 * R * (R + eps) / d ** 2, 2 * P * (P + eps) / d ** 2
    want = (1 + beta * k * y) * (u - y) / N - u * (1 - u) * (FP * dP + FR * dR)
    assert torch.allclose(z.grad, want, atol=1e-10)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        LS.loss("nope", {"logit": torch.zeros(1, 1, 2, 2), "u": torch.zeros(1, 1, 2, 2)}, torch.zeros(1, 1, 2, 2), 1)


def test_probability_input_path_clamps():
    y = torch.ones(1, 1, 4, 4)
    u = torch.zeros(1, 1, 4, 4)
    val = LS.bce({"u": u}, y, on_logit=False)
    assert torch.isfinite(val) and abs(float(val) - float(-torch.log(torch.tensor(1e-4)))) < 1e-4
