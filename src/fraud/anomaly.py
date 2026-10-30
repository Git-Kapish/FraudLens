"""Unsupervised scorers (M5 Isolation Forest, M6 autoencoder) and the rank-average ensemble."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from fraud.config import load_config
from fraud.models import score

logger = logging.getLogger(__name__)


class IsolationForestScorer:
    """M5: isolation forest fit on train; score is the negative decision function."""

    def __init__(self) -> None:
        cfg = load_config()
        self.model = IsolationForest(
            random_state=cfg["seed"], n_jobs=-1, **cfg["models"]["isolation_forest"]
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> IsolationForestScorer:
        """Fit without labels (y is accepted only for a uniform interface)."""
        self.model.fit(X)
        return self

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher means more anomalous."""
        return -self.model.decision_function(X)


def _build_net(n_features: int, hidden: list[int]) -> nn.Sequential:
    """Symmetric MLP autoencoder, e.g. 30-16-8-16-30 for hidden=[16, 8]."""
    sizes = [n_features, *hidden, *reversed(hidden[:-1]), n_features]
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class AutoencoderScorer:
    """M6: autoencoder trained on legit-only train rows; score is reconstruction error."""

    def __init__(self) -> None:
        self.cfg = load_config()["training"]["autoencoder"]
        self.net: nn.Sequential | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _standardise(self, X: pd.DataFrame) -> torch.Tensor:
        return torch.tensor((X.to_numpy() - self.mean_) / self.std_, dtype=torch.float32)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> AutoencoderScorer:
        """Train on rows with y == 0 only; scaling stats come from those rows too."""
        seed = load_config()["seed"]
        torch.manual_seed(seed)
        legit = X[np.asarray(y) == 0]
        self.mean_ = legit.to_numpy().mean(axis=0)
        self.std_ = np.where(legit.to_numpy().std(axis=0) == 0, 1.0, legit.to_numpy().std(axis=0))
        data = self._standardise(legit)
        self.net = _build_net(data.shape[1], self.cfg["hidden"])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(seed)
        for epoch in range(self.cfg["epochs"]):
            perm = torch.randperm(len(data), generator=gen)
            total = 0.0
            for i in range(0, len(data), self.cfg["batch_size"]):
                batch = data[perm[i : i + self.cfg["batch_size"]]]
                opt.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            logger.info(
                "AE epoch %d/%d loss %.4f", epoch + 1, self.cfg["epochs"], total / len(data)
            )
        return self

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Per-row mean squared reconstruction error (higher means more anomalous)."""
        data = self._standardise(X)
        self.net.eval()
        with torch.no_grad():
            return ((self.net(data) - data) ** 2).mean(dim=1).numpy()


class RankAverageEnsemble:
    """Average of each component's score percentile, measured against its train scores.

    Percentiles come from a stored sorted copy of the train scores, so any new row can
    be scored on its own (no batch ranking), and nothing is fit on val or test.
    """

    def __init__(self, components: list[Any]) -> None:
        self.components = components
        self.train_scores_: list[np.ndarray] = []

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray | None = None) -> RankAverageEnsemble:
        """Store each (already fitted) component's sorted train scores."""
        self.train_scores_ = [np.sort(score(c, X_train)) for c in self.components]
        return self

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """Mean percentile in [0, 1] across components."""
        pct = [
            np.searchsorted(ref, score(c, X), side="right") / len(ref)
            for c, ref in zip(self.components, self.train_scores_, strict=True)
        ]
        return np.mean(pct, axis=0)
