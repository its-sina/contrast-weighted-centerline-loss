import numpy as np
import torch

from segloss import analysis as A
from segloss import losses as LS
from segloss.metrics import CenterlineReference, merged_components, stratified_recall
from segloss.operating_point import fp_at, match
from segloss.stats import group_bootstrap


def _line_ref(tol=2, um=10.0):
    lab = np.zeros((50, 50), bool)
    lab[25, 5:45] = True
    contrast = np.zeros((50, 50))
    contrast[:, :25] = -1.0                 # left half of the line is low-contrast
    return lab, CenterlineReference(lab, contrast, 0.0, tol, um, full_width=False)


def test_shift_within_and_beyond_tolerance():
    lab, ref = _line_ref(tol=2)
    S = np.zeros_like(lab)
    S[27, 5:45] = True                       # shifted 2 px: within tolerance
    r = ref.evaluate(S)
    assert r["cl_recall"] == 1.0 and r["cl_precision"] == 1.0 and r["fp_burden"] == 0.0
    S = np.zeros_like(lab)
    S[28, 5:45] = True                       # shifted 3 px: outside tolerance
    r = ref.evaluate(S)
    assert r["cl_recall"] == 0.0 and r["cl_precision"] == 0.0


def test_low_high_split_and_fp_burden_units():
    lab, ref = _line_ref(tol=2, um=10.0)
    S = np.zeros_like(lab)
    S[25, 25:45] = True                      # only the high-contrast half
    S[5, 10:30] = True                       # a 20-px off-label segment
    r = ref.evaluate(S)
    assert r["lc_recall"] < 0.2 and r["hc_recall"] == 1.0
    n_off = int((r["fp_burden"] * (50 * 50 * (10 / 1000) ** 2)) / (10 / 1000) + 0.5)
    assert n_off == 20


def test_stratified_recall_bins():
    lab, ref = _line_ref()
    contrast = np.tile(np.linspace(-1, 1, 50), (50, 1))
    out = stratified_recall(ref, contrast, np.array([0.0]), lab)
    assert set(out) == {"strat0", "strat1"} and out["strat0"] == out["strat1"] == 1.0


def test_merged_components():
    raw = np.zeros((10, 10), bool)
    raw[2, 1:4] = raw[2, 6:9] = True
    bridged = raw.copy()
    bridged[2, 4:6] = True
    assert merged_components(raw, raw) == 0 and merged_components(raw, bridged) == 2


def test_match_meets_target():
    rng = np.random.default_rng(0)
    lab, _ = _line_ref()
    refs, maps = {}, {}
    for i in range(3):
        refs[i] = CenterlineReference(lab, np.zeros(lab.shape), 0.0, 2, 10.0, False)
        m = rng.uniform(0, 1, lab.shape) * 0.9
        m[lab] = 0.95
        maps[i] = m
    base = fp_at(maps, refs, 0.5)
    th, ok = match(maps, refs, 0.25 * base)
    assert ok and fp_at(maps, refs, th) <= 0.25 * base and th > 0.5
    th0, ok0 = match(maps, refs, -1.0)
    assert not ok0 and th0 == 1 - 1e-6


def test_group_bootstrap_estimand():
    rows = []
    for g, (n, d) in enumerate([(1, 0.1), (3, 0.3), (2, -0.1)]):
        for j in range(n):
            rows.append({"case": f"{g}-{j}", "group": f"g{g}", "cfg": "A", "op": "1", "m": 1.0 + d})
            rows.append({"case": f"{g}-{j}", "group": f"g{g}", "cfg": "B", "op": "1", "m": 1.0})
    b = group_bootstrap(rows, "A", "B", "m", op="1", n_boot=2000)
    assert b["n_groups"] == 3 and b["n_images"] == 6
    assert abs(b["mean_diff"] - (0.1 + 0.3 - 0.1) / 3) < 1e-12          # mean of group means, not of images
    assert b["ci95"][0] <= b["mean_diff"] <= b["ci95"][1]
    assert group_bootstrap(rows, "A", "C", "m", op="1") == {"n_groups": 0}


def test_closed_form_gradient_matches_autograd():
    torch.manual_seed(1)
    for r in (1, 2, 3):
        z = torch.randn(3, 1, 20, 20, dtype=torch.float64, requires_grad=True)
        y = (torch.rand(3, 1, 20, 20) > 0.8).double()
        omega = 1 + 2.0 * (torch.rand(3, 1, 20, 20) > 0.5).double()
        L, _ = LS.loss("centerline", {"logit": z, "u": torch.sigmoid(z)}, y, r, omega)
        L.backward()
        g = A.ctf_logit_gradient(z.detach(), y, r, omega)
        assert torch.allclose(z.grad, g["grad"], atol=1e-12)
        assert float(g["routed"].sum()) == float(y.sum())               # every label pixel routes exactly once


def test_recall_weights_reallocate():
    w0, w2 = A.recall_weights(1, 2, 0.0), A.recall_weights(1, 2, 2.0)
    assert abs(w0["low_share"] - 1 / 3) < 1e-12 and abs(w2["low_share"] - 3 / 5) < 1e-12
    assert w2["per_low"] > w0["per_low"] and w2["per_high"] < w0["per_high"]


def test_evaluation_recall_never_exceeds_training_recall():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lab = np.zeros((40, 40), bool)
        lab[rng.integers(5, 35), 3:37] = True
        lab[3:37, rng.integers(5, 35)] = True
        low = rng.uniform(size=lab.shape) < 0.5
        S = rng.uniform(size=lab.shape) < 0.08
        for tol in (1, 2, 3):
            out = A.surrogate_upper_bound(S, lab, low, tol)
            assert out["eval_recall"] <= out["train_recall"] + 1e-12


def test_window_corner_makes_bound_strict():
    lab = np.zeros((11, 11), bool)
    lab[5, 5] = True
    S = np.zeros_like(lab)
    S[7, 7] = True                           # Chebyshev distance 2, Euclidean 2.83
    out = A.surrogate_upper_bound(S, lab, np.ones_like(lab), 2)
    assert out["train_recall"] == 1.0 and out["eval_recall"] == 0.0
