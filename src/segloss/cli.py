"""Command-line demo: train a small U-Net with the CTF loss on synthetic thin curves (CPU, under a minute)."""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from segloss import contrast as C
from segloss.losses import tol_f1
from segloss.metrics import CenterlineReference
from segloss.models.unet import UNet
from segloss.synthetic import synthetic_set
from segloss.training import Crops, TrainConfig, forward_scores, make_optimizer, normalize, train_step
from segloss.variational import VariationalSegNet


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="segloss-demo",
        description="Generate synthetic thin curves, fit the low-contrast threshold, and run forward/backward "
                    "training steps of a small U-Net with the contrast-weighted tolerance-F1 (CTF) loss.")
    ap.add_argument("--loss", choices=["centerline", "dice_bce"], default="centerline",
                    help="'centerline' = CTF loss (default); 'dice_bce' = BCE + soft Dice reference")
    ap.add_argument("--head", choices=["none", "tv_iso", "tv_dir", "full"], default="none",
                    help="optional end-to-end TV layer after the U-Net (default none)")
    ap.add_argument("--tv-iters", type=int, default=30, help="unrolled TV iterations K (default 30)")
    ap.add_argument("--steps", type=int, default=30, help="training steps (default 30)")
    ap.add_argument("--images", type=int, default=6, help="number of synthetic training images (default 6)")
    ap.add_argument("--size", type=int, default=128, help="synthetic image side in pixels (default 128)")
    ap.add_argument("--crop", type=int, default=64, help="training crop side in pixels (default 64)")
    ap.add_argument("--batch", type=int, default=4, help="batch size (default 4)")
    ap.add_argument("--radius", type=int, default=2, help="tolerance radius r in pixels (default 2)")
    ap.add_argument("--beta", type=float, default=2.0, help="contrast weight beta; 0 disables it (default 2)")
    ap.add_argument("--base-filters", type=int, default=8, help="U-Net base filters (default 8)")
    ap.add_argument("--depth", type=int, default=3, help="U-Net depth (default 3)")
    ap.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate (default 1e-3)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    ap.add_argument("--log-every", type=int, default=5, help="print every N steps (default 5)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    torch.manual_seed(a.seed)
    t0 = time.time()
    imgs, labs = synthetic_set(a.images + 1, a.size, seed=a.seed)
    train_imgs, train_labs = imgs[:-1], labs[:-1]
    c_thr = C.fit_low_contrast_threshold(train_imgs, train_labs)
    lowc = [C.low_contrast_indicator(i, c_thr) for i in train_imgs]
    frac = float(np.mean([k[l].mean() for k, l in zip(lowc, train_labs)]))
    print(f"synthetic set: {a.images} train + 1 unseen image, {a.size}x{a.size}; c_thr = {c_thr:.3f}; "
          f"low-contrast fraction of label pixels = {frac:.2f}")

    cfg = TrainConfig(loss=a.loss, beta=a.beta, radius=a.radius, lr=a.lr, warmup=max(1, a.steps // 10),
                      iters=a.steps)
    if a.head == "none":
        net = UNet(1, 1, base_filters=a.base_filters, depth=a.depth)
    else:
        net = VariationalSegNet({"mode": a.head, "base_filters": a.base_filters, "depth": a.depth,
                                 "var": {"K": a.tv_iters}, "compile": False})
    opt, sched = make_optimizer(net, cfg)
    ds = Crops([normalize(i) for i in train_imgs], train_labs, lowc, a.crop, a.steps * a.batch, seed=(a.seed, 0))
    net.train()
    for step, batch in enumerate(DataLoader(ds, batch_size=a.batch), start=1):
        out = train_step(net, opt, sched, batch, cfg)
        if step % a.log_every == 0 or step == 1:
            extra = f"  P={out['P']:.3f} R={out['R']:.3f}" if "P" in out else ""
            print(f"step {step:4d}  loss={out['loss']:.4f}  grad_norm={out['grad_norm']:.3f}{extra}")

    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(normalize(imgs[-1]))[None]
        f = net(x) if a.head != "none" else forward_scores(net, x)
        u = f["u"]
        y = torch.from_numpy(labs[-1].astype(np.float32))[None, None]
        k = torch.from_numpy(C.low_contrast_indicator(imgs[-1], c_thr))[None, None]
        F_all, P, R = tol_f1(u, y, a.radius)
        _, _, R_lc = tol_f1(u, y * k, a.radius)
    print(f"unseen synthetic image: soft tolerance P={float(P):.3f} R={float(R):.3f} F1={float(F_all):.3f}; "
          f"low-contrast soft recall={float(R_lc):.3f}")
    ref = CenterlineReference(labs[-1], C.contrast_map(imgs[-1]), c_thr, a.radius, 10.0, full_width=False)
    m = ref.evaluate(u[0, 0].numpy() > 0.5)
    print(f"  at threshold 0.5 (skeleton, Euclidean tolerance {a.radius}px): cl-P={m['cl_precision']:.3f} "
          f"cl-R={m['cl_recall']:.3f} LC-R={m['lc_recall']:.3f} HC-R={m['hc_recall']:.3f}")
    if a.head != "none":
        print(f"  TV layer: mean lambda={float(f['lam'].mean()):.4f}  mean |u - p|={float((f['u'] - f['p']).abs().mean()):.4f}")
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
