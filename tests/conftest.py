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


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()
