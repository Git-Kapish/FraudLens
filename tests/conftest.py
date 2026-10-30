"""Shared synthetic fixture. Real data is never used in unit tests."""

import numpy as np
import pandas as pd
import pytest

from fraud.data import FEATURES, TARGET


def make_frame(n: int = 4000, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the real schema; frauds are spread over time."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["Time"] = np.sort(rng.integers(0, 172_800, size=n)).astype(float)
    df["Amount"] = rng.gamma(2.0, 30.0, size=n)
    df[TARGET] = (rng.random(n) < fraud_rate).astype(int)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffled input


def make_artifacts(frame: pd.DataFrame, threshold: float = 0.3):
    """API artifacts around a tiny synthetic model, so tests need neither data nor models/."""
    from xgboost import XGBClassifier

    from fraud.api.main import Artifacts
    from fraud.calibrate import CalibratedModel, fit_isotonic
    from fraud.explain import build_explainer
    from fraud.features import FeaturePipeline

    pipeline = FeaturePipeline().fit(frame)
    X, y = pipeline.transform(frame), frame[TARGET].to_numpy()
    base = XGBClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
    model = CalibratedModel(base, fit_isotonic(base.predict_proba(X)[:, 1], y))
    return Artifacts(pipeline, model, build_explainer(model), threshold, "test-model")


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()
