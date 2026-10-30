import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud.calibrate import CalibratedModel, fit_isotonic, oof_scores


def _xy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = (X["a"] + rng.normal(scale=0.5, size=n) > 1).astype(int).to_numpy()
    return X, y


def test_oof_scores_only_cover_later_blocks_with_matching_labels():
    X, y = _xy()
    scores, labels = oof_scores(lambda yf: LogisticRegression(), X, y, n_splits=5)
    # TimeSeriesSplit(5) on 600 rows: the first 100 rows are never a test fold
    assert len(scores) == 500
    assert np.array_equal(labels, y[100:])


def test_oof_fold_models_never_see_future_rows():
    X, y = _xy()
    seen = []

    class Spy(LogisticRegression):
        def fit(self, X, y, *a, **k):
            seen.append(X.index.max())
            return super().fit(X, y, *a, **k)

        def predict_proba(self, X):
            assert X.index.min() > seen[-1]
            return super().predict_proba(X)

    oof_scores(lambda yf: Spy(), X, y, n_splits=5)
    assert len(seen) == 5


def test_isotonic_is_monotone_and_probabilities_valid():
    X, y = _xy()
    base = LogisticRegression().fit(X, y)
    raw = base.predict_proba(X)[:, 1]
    model = CalibratedModel(base, fit_isotonic(raw, y))
    p = model.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert ((0 <= p) & (p <= 1)).all()
    order = np.argsort(raw)
    assert (np.diff(p[order, 1]) >= -1e-12).all()
