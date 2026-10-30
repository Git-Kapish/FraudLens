"""Optuna tuning of XGBoost, scored by PR-AUC on time-ordered CV folds over train only."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from fraud.config import load_config
from fraud.models import build_xgb, scale_pos_weight

logger = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample one XGBoost parameter set from the config search space."""
    params: dict[str, Any] = {}
    for name, s in space.items():
        if s.get("type") == "int":
            params[name] = trial.suggest_int(name, s["low"], s["high"])
        else:
            params[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
    return params


def cv_pr_auc(params: dict[str, Any], X: pd.DataFrame, y: np.ndarray, n_splits: int) -> float:
    """Mean PR-AUC over TimeSeriesSplit folds; each fold trains only on earlier rows."""
    scores = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        if y[te].sum() == 0:
            raise ValueError("A CV fold has no frauds; lower cv_splits")
        model = build_xgb(params, spw=scale_pos_weight(y[tr]))
        model.fit(X.iloc[tr], y[tr])
        scores.append(average_precision_score(y[te], model.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(scores))


def tune_xgb(X: pd.DataFrame, y: np.ndarray) -> optuna.Study:
    """Run the study (seeded TPE sampler) and return it."""
    cfg = load_config()
    train_cfg = cfg["training"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        value = cv_pr_auc(
            suggest_params(trial, train_cfg["search_space"]), X, y, train_cfg["cv_splits"]
        )
        logger.info("trial %d: CV PR-AUC %.4f", trial.number, value)
        return value

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"])
    )
    study.optimize(objective, n_trials=train_cfg["optuna_trials"])
    logger.info("best CV PR-AUC %.4f with %s", study.best_value, study.best_params)
    return study
