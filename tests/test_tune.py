import numpy as np
import pytest
from conftest import make_frame
from optuna.trial import FixedTrial

from fraud.config import load_config
from fraud.features import FeaturePipeline
from fraud.tune import cv_pr_auc, suggest_params

SPACE = load_config()["training"]["search_space"]


def _midpoint(spec):
    mid = (spec["low"] + spec["high"]) / 2
    return int(mid) if spec.get("type") == "int" else mid


def test_suggest_params_covers_the_search_space_within_bounds():
    trial = FixedTrial({name: _midpoint(spec) for name, spec in SPACE.items()})
    params = suggest_params(trial, SPACE)
    assert set(params) == set(SPACE)
    for name, value in params.items():
        assert SPACE[name]["low"] <= value <= SPACE[name]["high"]
        if SPACE[name].get("type") == "int":
            assert isinstance(value, int)


def _sorted_xy(n=1800):
    frame = make_frame(n=n, fraud_rate=0.1).sort_values("Time").reset_index(drop=True)
    return FeaturePipeline().fit(frame).transform(frame), frame["Class"].to_numpy()


def test_cv_pr_auc_returns_a_valid_average_precision():
    X, y = _sorted_xy()
    score = cv_pr_auc({"n_estimators": 10, "max_depth": 3}, X, y, n_splits=3)
    assert 0.0 < score <= 1.0


def test_cv_raises_when_a_fold_has_no_frauds():
    X, y = _sorted_xy()
    y = np.zeros_like(y)
    y[0] = 1  # the only fraud is in the first training block, so every test fold is empty
    with pytest.raises(ValueError, match="no frauds"):
        cv_pr_auc({"n_estimators": 5}, X, y, n_splits=3)
