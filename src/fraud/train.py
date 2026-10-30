"""Train models on the train split, score on validation, log everything to MLflow.

The test split is never loaded into a model here; it is only evaluated in evaluate.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.config import load_config, seed_everything
from fraud.data import load_splits, xy
from fraud.features import FeaturePipeline
from fraud.models import MODEL_NAMES, build_model, score

logger = logging.getLogger(__name__)


def build_matrices() -> tuple[FeaturePipeline, dict[str, tuple[pd.DataFrame, np.ndarray]]]:
    """Fit the feature pipeline on train and transform train and val (test is left alone)."""
    splits = load_splits()
    X_train, y_train = xy(splits["train"])
    pipeline = FeaturePipeline().fit(X_train)
    data = {"train": (pipeline.transform(X_train), y_train.to_numpy())}
    X_val, y_val = xy(splits["val"])
    data["val"] = (pipeline.transform(X_val), y_val.to_numpy())
    return pipeline, data


def val_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    """Validation PR-AUC (headline) and ROC-AUC (secondary)."""
    return {"val_pr_auc": average_precision_score(y, s), "val_roc_auc": roc_auc_score(y, s)}


def log_run(name: str, model: Any, params: dict[str, Any], metrics: dict[str, float]) -> None:
    """Log one model's params, metrics and fitted artifact as an MLflow run."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # our own freshly trained models, so cloudpickle is fine (skops rejects sklearn Tree)
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")


def fit_supervised(
    data: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit M0-M4 on train, score on val. Returns fitted models and val PR-AUC by name."""
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    models, pr_auc = {}, {}
    for name in MODEL_NAMES:
        model = build_model(name, y_train)
        model.fit(X_train, y_train)
        metrics = val_metrics(y_val, score(model, X_val))
        log_run(name, model, {"model": MODEL_NAMES[name], "seed": load_config()["seed"]}, metrics)
        models[name], pr_auc[name] = model, metrics["val_pr_auc"]
        logger.info("%s: val PR-AUC %.4f", name, pr_auc[name])
    return models, pr_auc


def init_mlflow() -> None:
    """Point MLflow at the local file store from config.

    MLflow 3.x marks the file store as maintenance-only and refuses it unless opted in.
    The PRD asks for a local file store, so we opt in (see reports/decisions.md).
    """
    cfg = load_config()["mlflow"]
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed_everything()
    init_mlflow()
    _, data = build_matrices()
    _, pr_auc = fit_supervised(data)
    table = pd.Series({MODEL_NAMES[k]: v for k, v in pr_auc.items()}, name="val PR-AUC")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
