import numpy as np
import pandas as pd
import pytest

from fraud.anomaly import AutoencoderScorer, IsolationForestScorer, RankAverageEnsemble
from fraud.features import MODEL_COLUMNS


@pytest.fixture
def blobs():
    """2000 legit rows ~ N(0, 1) and 20 anomalies ~ N(6, 1), in the model-matrix schema."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (2000, 30)), rng.normal(6, 1, (20, 30))])
    y = np.r_[np.zeros(2000), np.ones(20)].astype(int)
    return pd.DataFrame(X, columns=MODEL_COLUMNS), y


def test_autoencoder_trains_on_legit_rows_only(blobs):
    X, y = blobs
    ae = AutoencoderScorer().fit(X, y)
    assert np.allclose(ae.mean_, X[y == 0].mean().to_numpy())
    s = ae.raw_score(X)
    assert s[y == 1].mean() > 5 * s[y == 0].mean()


def test_autoencoder_architecture_is_30_16_8_16_30(blobs):
    X, y = blobs
    ae = AutoencoderScorer().fit(X, y)
    widths = [(m.in_features, m.out_features) for m in ae.net if hasattr(m, "in_features")]
    assert widths == [(30, 16), (16, 8), (8, 16), (16, 30)]


def test_isolation_forest_scores_outliers_higher(blobs):
    X, y = blobs
    s = IsolationForestScorer().fit(X).raw_score(X)
    assert s[y == 1].mean() > s[y == 0].mean()


def test_rank_average_uses_train_scores_only(blobs):
    X, y = blobs
    train = X.iloc[:1000]
    iso = IsolationForestScorer().fit(train)
    ens = RankAverageEnsemble([iso, iso]).fit(train)
    assert all(len(ref) == len(train) for ref in ens.train_scores_)
    s = ens.raw_score(X)
    assert ((0 <= s) & (s <= 1)).all()
    assert s[y == 1].mean() > s[y == 0].mean()
    # a single row scores the same alone as inside a batch (no batch ranking)
    assert ens.raw_score(X.iloc[[5]])[0] == pytest.approx(s[5])
