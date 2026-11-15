"""Paired, group-level bootstrap of a difference between two methods.

    Delta = (1/|G|) sum_g (1/|I_g|) sum_{j in I_g} (mu_A(j) - mu_B(j))

Per-image differences are averaged within groups (e.g. images of the same subject), and the groups are resampled
with replacement; the interval is the 2.5 / 97.5 percentile interval of the resampled means. Rows are dicts with
keys ``case``, ``group``, ``cfg``, ``op`` and the metric. Non-finite differences are dropped. An interval that
contains zero does not establish equivalence.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import List, Optional

import numpy as np

__all__ = ["load_rows", "group_bootstrap"]


def load_rows(path) -> List[dict]:
    return [json.loads(l) for l in open(path)]


def group_bootstrap(rows: List[dict], cfg_a: str, cfg_b: str, metric: str, op: str = "1.25",
                    op_b: Optional[str] = None, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired difference a - b of `metric`: per-image differences averaged within groups, bootstrap over groups."""
    op_b = op_b or op
    A = {r["case"]: r for r in rows if r["cfg"] == cfg_a and r["op"] == op}
    B = {r["case"]: r for r in rows if r["cfg"] == cfg_b and r["op"] == op_b}
    per_g = defaultdict(list)
    for c in sorted(set(A) & set(B)):
        d = A[c][metric] - B[c][metric]
        if np.isfinite(d):
            per_g[A[c]["group"]].append(d)
    g = np.array([np.mean(v) for _, v in sorted(per_g.items())])
    if len(g) == 0:
        return {"n_groups": 0}
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n_boot, len(g)))].mean(1)
    return {"n_groups": int(len(g)), "n_images": int(sum(len(v) for v in per_g.values())), "mean_diff": float(g.mean()),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "frac_groups_positive": float((g > 0).mean())}
