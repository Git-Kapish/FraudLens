"""Train all models on the train split, choose on validation, save frozen artifacts.

The test split is never loaded here; it is only evaluated once, in evaluate.py.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fraud.anomaly import AutoencoderScorer, IsolationForestScorer, RankAverageEnsemble
from fraud.calibrate import CalibratedModel, fit_isotonic, oof_scores, plot_reliability
from fraud.config import load_config, resolve_path, seed_everything
from fraud.data import FEATURES, load_splits, xy
from fraud.features import MODEL_COLUMNS, FeaturePipeline
from fraud.models import (
    DISPLAY_NAMES,
    MODEL_NAMES,
    build_model,
    build_xgb,
    scale_pos_weight,
    score,
)
from fraud.tune import tune_xgb

logger = logging.getLogger(__name__)
Data = dict[str, tuple[pd.DataFrame, np.ndarray]]


def build_matrices() -> tuple[FeaturePipeline, Data]:
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


def init_mlflow() -> None:
    """Point MLflow at the local file store from config.

    MLflow 3.x marks the file store as maintenance-only and refuses it unless opted in.
    The PRD asks for a local file store, so we opt in (see reports/decisions.md).
    """
    cfg = load_config()["mlflow"]
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"])


def log_run(name: str, model: Any, params: dict[str, Any], metrics: dict[str, float]) -> None:
    """Log one model's params, metrics and fitted artifact as an MLflow run."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if isinstance(model, BaseEstimator):
            # our own freshly trained models, so cloudpickle is fine (skops rejects sklearn Tree)
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.joblib"
                joblib.dump(model, path)
                mlflow.log_artifact(str(path))


def evaluate_and_log(
    name: str, model: Any, data: Data, params: dict[str, Any] | None = None
) -> float:
    """Score a fitted model on val, log the run, return val PR-AUC."""
    X_val, y_val = data["val"]
    metrics = val_metrics(y_val, score(model, X_val))
    log_run(name, model, {"model": DISPLAY_NAMES[name], **(params or {})}, metrics)
    logger.info("%s: val PR-AUC %.4f", name, metrics["val_pr_auc"])
    return metrics["val_pr_auc"]


def fit_supervised(data: Data, models: dict[str, Any], pr_auc: dict[str, float]) -> None:
    """M0-M4: fit on train, score on val."""
    X_train, y_train = data["train"]
    for name in MODEL_NAMES:
        models[name] = build_model(name, y_train).fit(X_train, y_train)
        pr_auc[name] = evaluate_and_log(name, models[name], data)


def fit_anomaly(data: Data, models: dict[str, Any], pr_auc: dict[str, float]) -> None:
    """M5 Isolation Forest (unlabeled train) and M6 autoencoder (legit-only train rows)."""
    X_train, y_train = data["train"]
    models["M5_iforest"] = IsolationForestScorer().fit(X_train)
    pr_auc["M5_iforest"] = evaluate_and_log("M5_iforest", models["M5_iforest"], data)
    ae_cfg = load_config()["training"]["autoencoder"]
    models["M6_autoencoder"] = AutoencoderScorer().fit(X_train, y_train)
    pr_auc["M6_autoencoder"] = evaluate_and_log(
        "M6_autoencoder",
        models["M6_autoencoder"],
        data,
        {f"ae_{k}": str(v) for k, v in ae_cfg.items()},
    )


def fit_tuned(data: Data, models: dict[str, Any], pr_auc: dict[str, float]) -> dict[str, Any]:
    """Optuna-tune XGBoost on train CV, refit the best params on train, score on val."""
    X_train, y_train = data["train"]
    study = tune_xgb(X_train, y_train)
    models["M3_tuned"] = build_xgb(study.best_params, spw=scale_pos_weight(y_train))
    models["M3_tuned"].fit(X_train, y_train)
    pr_auc["M3_tuned"] = evaluate_and_log(
        "M3_tuned",
        models["M3_tuned"],
        data,
        {
            **study.best_params,
            "cv_pr_auc": round(study.best_value, 4),
            "n_trials": len(study.trials),
        },
    )
    return study.best_params


def fit_ensemble(data: Data, models: dict[str, Any], pr_auc: dict[str, float]) -> None:
    """Optional M7: rank-average of tuned XGBoost and the autoencoder."""
    X_train, y_train = data["train"]
    ensemble = RankAverageEnsemble([models["M3_tuned"], models["M6_autoencoder"]])
    models["M7_ensemble"] = ensemble.fit(X_train)
    pr_auc["M7_ensemble"] = evaluate_and_log("M7_ensemble", ensemble, data)


def pick_deployable(pr_auc: dict[str, float]) -> str:
    """Best-by-validation among the XGBoost-family models allowed for serving."""
    allowed = load_config()["training"]["deployable_models"]
    return max(allowed, key=lambda n: pr_auc[n])


def calibrate_best(
    name: str, base: Any, best_params: dict[str, Any], data: Data
) -> tuple[CalibratedModel, dict[str, float]]:
    """Fit isotonic on out-of-fold train scores of `name`'s recipe; check on val."""
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    factories: dict[str, Callable[[np.ndarray], Any]] = {
        "M3_xgb": lambda yf: build_xgb(spw=scale_pos_weight(yf)),
        "M3_tuned": lambda yf: build_xgb(best_params, spw=scale_pos_weight(yf)),
        "M4_xgb_smote": lambda yf: build_model("M4_xgb_smote", yf),
    }
    scores, labels = oof_scores(
        factories[name], X_train, y_train, load_config()["training"]["cv_splits"]
    )
    calibrated = CalibratedModel(base, fit_isotonic(scores, labels))
    raw_val = base.predict_proba(X_val)[:, 1]
    cal_val = calibrated.predict_proba(X_val)[:, 1]
    plot_reliability(y_val, raw_val, cal_val)
    metrics = {
        **val_metrics(y_val, cal_val),
        "val_brier_raw": brier_score_loss(y_val, raw_val),
        "val_brier_calibrated": brier_score_loss(y_val, cal_val),
    }
    log_run(
        "best_calibrated", calibrated, {"base": name, "method": "isotonic (OOF train)"}, metrics
    )
    logger.info("calibrated %s: %s", name, {k: round(v, 5) for k, v in metrics.items()})
    return calibrated, metrics


def save_artifacts(
    pipeline: FeaturePipeline,
    models: dict[str, Any],
    calibrated: CalibratedModel,
    pr_auc: dict[str, float],
    best: str,
    best_params: dict[str, Any],
    cal_metrics: dict[str, float],
) -> Path:
    """Write fitted objects (gitignored) and metadata.json (tracked) to models/."""
    out = resolve_path("models_dir")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "pipeline.joblib")
    joblib.dump(calibrated, out / "best_calibrated.joblib")
    for name, model in models.items():
        joblib.dump(model, out / f"{name}.joblib")
    meta = {
        "model_version": f"{best}-{date.today():%Y%m%d}",
        "seed": load_config()["seed"],
        "best_model": best,
        "calibration": "isotonic, fit on out-of-fold train predictions",
        "raw_features": FEATURES,
        "model_features": MODEL_COLUMNS,
        "models": list(models),
        "val_pr_auc": {k: round(v, 4) for k, v in pr_auc.items()},
        "calibrated_val": {k: round(v, 6) for k, v in cal_metrics.items()},
        "tuned_params": best_params,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed_everything()
    init_mlflow()
    pipeline, data = build_matrices()
    models: dict[str, Any] = {}
    pr_auc: dict[str, float] = {}
    fit_supervised(data, models, pr_auc)
    fit_anomaly(data, models, pr_auc)
    best_params = fit_tuned(data, models, pr_auc)
    fit_ensemble(data, models, pr_auc)
    best = pick_deployable(pr_auc)
    calibrated, cal_metrics = calibrate_best(best, models[best], best_params, data)
    out = save_artifacts(pipeline, models, calibrated, pr_auc, best, best_params, cal_metrics)
    table = pd.Series({DISPLAY_NAMES[k]: v for k, v in pr_auc.items()}, name="val PR-AUC")
    print(table.round(4).to_string())
    print(f"\nDeployed base model: {DISPLAY_NAMES[best]} (calibrated). Saved to {out}")


if __name__ == "__main__":
    main()
