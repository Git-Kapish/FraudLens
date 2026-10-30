"""Feature pipeline: Amount and Time derived features. Fit on train only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud.data import FEATURES

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
DERIVED_COLUMNS = ["amount_log", "hour"]
MODEL_COLUMNS = [*PCA_COLUMNS, *DERIVED_COLUMNS]


def hour_of_day(time: pd.Series) -> pd.Series:
    """Hour since the first transaction, wrapped to 0-23 (the real start clock time is unknown)."""
    return ((time // SECONDS_PER_HOUR) % HOURS_PER_DAY).astype(int)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Raw log1p(Amount) and hour-of-day, before scaling."""
    return pd.DataFrame(
        {"amount_log": np.log1p(df["Amount"]), "hour": hour_of_day(df["Time"])}, index=df.index
    )


class FeaturePipeline:
    """V1-V28 pass through; log1p(Amount) and hour are standardised with train statistics.

    Raw `Time` is dropped: it only grows, so a model trained on early rows would
    never see the values it is scored on.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, train: pd.DataFrame) -> FeaturePipeline:
        """Learn scaling statistics from the training frame only."""
        self.scaler.fit(_derive(train))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the model matrix (28 PCA columns + 2 scaled derived columns)."""
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        derived = pd.DataFrame(
            self.scaler.transform(_derive(df)), columns=DERIVED_COLUMNS, index=df.index
        )
        return pd.concat([df[PCA_COLUMNS], derived], axis=1)[MODEL_COLUMNS]
