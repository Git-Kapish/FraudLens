"""Isotonic calibration fit on out-of-fold train predictions (never on val or test)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from fraud.config import load_config, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)


def oof_scores(
    make_model: Callable[[np.ndarray], Any], X: pd.DataFrame, y: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities from time-ordered folds. Returns (scores, labels).

    `make_model(y_fold)` returns a fresh unfitted model. The earliest block of rows is
    never a test fold, so it is absent from the output.
    """
    scores, labels = [], []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = make_model(y[tr])
        model.fit(X.iloc[tr], y[tr])
        scores.append(model.predict_proba(X.iloc[te])[:, 1])
        labels.append(y[te])
    return np.concatenate(scores), np.concatenate(labels)


def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Monotone map from raw score to fraud probability."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y)


class CalibratedModel:
    """A fitted model plus its isotonic calibrator; exposes calibrated `predict_proba`."""

    def __init__(self, base: Any, calibrator: IsotonicRegression) -> None:
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities as an (n, 2) array."""
        p = self.calibrator.predict(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1.0 - p, p])


def plot_reliability(
    y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, out_dir: Path | None = None
) -> Path:
    """Reliability curve (quantile bins, since frauds are rare) for raw vs calibrated scores."""
    bins = load_config()["training"]["calibration_bins"]
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, s in [("raw", raw), ("calibrated", calibrated)]:
        frac, mean = calibration_curve(y, s, n_bins=bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend()
    out_dir = out_dir or resolve_path("figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reliability_curve.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
