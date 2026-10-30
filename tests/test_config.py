import numpy as np
import torch

from fraud.calibrate import plot_reliability
from fraud.config import ROOT, load_config, resolve_path, seed_everything


def test_seed_everything_makes_numpy_and_torch_reproducible():
    seed_everything(7)
    a = (np.random.rand(3), torch.rand(3))
    seed_everything(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_default_seed_comes_from_config():
    assert seed_everything() == load_config()["seed"] == 42


def test_resolve_path_is_absolute_and_inside_the_repo():
    path = resolve_path("models_dir")
    assert path.is_absolute() and ROOT in path.parents


def test_reliability_plot_is_written(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.random(2000) ** 3
    y = (rng.random(2000) < raw).astype(int)
    path = plot_reliability(y, raw, raw, out_dir=tmp_path)
    assert path == tmp_path / "reliability_curve.png" and path.stat().st_size > 0
