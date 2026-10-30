import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud.evaluate import (
    bootstrap_ci,
    confusion,
    f_beta,
    operating_point,
    precision_at_recall,
    recall_at_precision,
)

Y = np.array([1, 1, 0, 0, 1])
S = np.array([0.9, 0.8, 0.7, 0.2, 0.1])


def test_confusion_and_operating_point():
    y, flagged = np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
    assert confusion(y, flagged) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    op = operating_point(y, flagged)
    assert op["precision"] == 0.5 and op["recall"] == 0.5 and op["f1"] == 0.5


def test_f_beta_hand_computed():
    assert f_beta(1.0, 0.5, 2.0) == pytest.approx(5 * 0.5 / 4.5)
    assert f_beta(0.0, 0.0, 1.0) == 0.0


def test_recall_at_precision_hand_computed():
    # top-2 flagged: precision 1.0, recall 2/3; adding row 3 drops precision to 2/3
    assert recall_at_precision(Y, S, 0.9) == pytest.approx(2 / 3)


def test_precision_at_recall_hand_computed():
    # recall >= 0.8 needs all three frauds, which means flagging all 5 rows: precision 3/5
    assert precision_at_recall(Y, S, 0.8) == pytest.approx(0.6)


def test_unreachable_targets_give_zero():
    y, s = np.array([0, 1]), np.array([0.9, 0.1])
    assert recall_at_precision(y, s, 0.99) == 0.0


def test_bootstrap_ci_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    s = y * 0.5 + rng.random(2000)
    kw = dict(metrics={"ap": average_precision_score}, n_resamples=200, level=0.95, seed=42)
    a, b = bootstrap_ci(y, s, **kw), bootstrap_ci(y, s, **kw)
    assert a == b
    lo, hi = a["ap"]
    assert lo < average_precision_score(y, s) < hi
