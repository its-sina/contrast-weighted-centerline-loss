import numpy as np
import torch
from torch.utils.data import DataLoader

from segloss import contrast as C
from segloss.cli import build_parser, main
from segloss.models.unet import UNet
from segloss.synthetic import synthetic_set
from segloss.training import Crops, TrainConfig, make_optimizer, normalize, train_step


def test_contrast_constant_image_is_finite_and_zero():
    c = C.contrast_map(np.full((32, 32), 5.0))
    assert np.isfinite(c).all() and np.allclose(c, 0)


def test_bright_line_has_higher_contrast_than_faint_line():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 0.1, (64, 64))
    img[20, :] += 2.0
    img[44, :] += 0.3
    c = C.contrast_map(img)
    assert np.median(c[20]) > np.median(c[44]) > np.median(c[32])


def test_threshold_is_one_third_quantile_at_skeleton_pixels():
    imgs, labs = synthetic_set(3, 64, seed=1)
    thr = C.fit_low_contrast_threshold(imgs, labs)
    vals = np.concatenate([C.contrast_map(i)[C.skeleton(l)] for i, l in zip(imgs, labs)])
    assert abs(np.mean(vals < thr) - 1 / 3) < 0.02


def test_crops_transform_label_and_indicator_together():
    img = np.zeros((40, 40), np.float32)
    lab = np.zeros((40, 40), bool)
    lab[10, 5:30] = True
    low = lab.astype(np.float32)
    ds = Crops([normalize(img)], [lab], [low], 32, 20, seed=(0, 0))
    for k in range(len(ds)):
        x, y, c = ds[k]
        assert x.shape == y.shape == c.shape == (1, 32, 32)
        assert torch.equal(y, c)


def test_train_step_runs_and_updates_parameters():
    imgs, labs = synthetic_set(2, 64, seed=2)
    thr = C.fit_low_contrast_threshold(imgs, labs)
    low = [C.low_contrast_indicator(i, thr) for i in imgs]
    cfg = TrainConfig(radius=1, warmup=1, iters=3)
    torch.manual_seed(0)
    net = UNet(1, 1, base_filters=4, depth=2)
    opt, sched = make_optimizer(net, cfg)
    before = [p.detach().clone() for p in net.parameters()]
    batch = next(iter(DataLoader(Crops([normalize(i) for i in imgs], labs, low, 32, 2, (0, 0)), batch_size=2)))
    out = train_step(net, opt, sched, batch, cfg)
    assert np.isfinite(out["loss"]) and "P" in out and "R" in out
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))


def test_cli_help_and_tiny_run(capsys):
    assert build_parser().format_help().startswith("usage: segloss-demo")
    assert main(["--steps", "2", "--images", "2", "--size", "64", "--crop", "32", "--batch", "2",
                 "--base-filters", "4", "--depth", "2", "--log-every", "1"]) == 0
    assert "unseen synthetic image" in capsys.readouterr().out
