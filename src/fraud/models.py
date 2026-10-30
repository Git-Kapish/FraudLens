"""Model factory for the supervised comparison (M0-M4)."""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud.config import load_config

MODEL_NAMES = {
    "M0_dummy": "Dummy (prior)",
    "M1_logreg": "Logistic Regression (balanced)",
    "M2_rf": "Random Forest (balanced_subsample)",
    "M3_xgb": "XGBoost (scale_pos_weight)",
    "M4_xgb_smote": "XGBoost + SMOTE (train only)",
}


def scale_pos_weight(y: np.ndarray) -> float:
    """Negatives per positive, the XGBoost class weight."""
    pos = float(np.sum(y == 1))
    return float(np.sum(y == 0)) / pos


def build_xgb(params: dict[str, Any] | None = None, spw: float | None = None) -> XGBClassifier:
    """XGBoost classifier with config defaults, overridden by `params`."""
    cfg = load_config()
    kwargs = {
        **cfg["models"]["xgboost"],
        "random_state": cfg["seed"],
        "eval_metric": "aucpr",
        "n_jobs": -1,
        "tree_method": "hist",
        **(params or {}),
    }
    if spw is not None:
        kwargs["scale_pos_weight"] = spw
    return XGBClassifier(**kwargs)


def build_model(name: str, y_train: np.ndarray) -> Any:
    """Return an unfitted model. `y_train` is only used for class-weight ratios."""
    cfg = load_config()
    seed = cfg["seed"]
    if name == "M0_dummy":
        return DummyClassifier(strategy="prior")
    if name == "M1_logreg":
        return LogisticRegression(
            class_weight="balanced", random_state=seed, **cfg["models"]["logreg"]
        )
    if name == "M2_rf":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
            **cfg["models"]["random_forest"],
        )
    if name == "M3_xgb":
        return build_xgb(spw=scale_pos_weight(y_train))
    if name == "M4_xgb_smote":
        smote = SMOTE(random_state=seed, **cfg["models"]["smote"])
        return Pipeline([("smote", smote), ("clf", build_xgb())])
    raise ValueError(f"Unknown model: {name}")


def score(model: Any, X: Any) -> np.ndarray:
    """Fraud probability for each row."""
    return model.predict_proba(X)[:, 1]
