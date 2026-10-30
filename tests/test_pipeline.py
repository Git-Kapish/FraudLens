"""End to end: `train.main()` then `evaluate.main()` on small synthetic data.

Runs the real pipeline wiring (all models, tuning, calibration, threshold, reports, SHAP)
with tiny settings, writing into a temp folder, so CI checks it without the dataset.
"""

import copy
import json

import pytest
from conftest import make_frame
from fastapi.testclient import TestClient

from fraud import evaluate, train
from fraud.api.main import create_app
from fraud.config import load_config
from fraud.data import FEATURES, TARGET

FIGURES = ["pr_curves.png", "roc_curves.png", "cost_vs_threshold.png", "reliability_curve.png"]


@pytest.fixture
def tiny_pipeline(tmp_path):
    """Shrink the shared config in place (every module reads the same cached dict), then restore."""
    cfg = load_config()
    saved = copy.deepcopy(cfg)
    frame = make_frame(n=6000, fraud_rate=0.05, seed=1)
    frame.loc[frame[TARGET] == 1, "V14"] -= 3.0  # a real signal, so some frauds get flagged
    raw = tmp_path / "raw" / "creditcard.csv"
    raw.parent.mkdir()
    frame.to_csv(raw, index=False)
    cfg["paths"].update(
        raw_data=str(raw),
        models_dir=str(tmp_path / "models"),
        figures_dir=str(tmp_path / "reports" / "figures"),
        results_md=str(tmp_path / "reports" / "results.md"),
    )
    cfg["mlflow"]["tracking_uri"] = (tmp_path / "mlruns").as_uri()
    cfg["models"]["random_forest"]["n_estimators"] = 10
    cfg["models"]["xgboost"]["n_estimators"] = 30
    cfg["models"]["isolation_forest"]["n_estimators"] = 20
    training = cfg["training"]
    training.update(optuna_trials=2, cv_splits=3)
    training["autoencoder"]["epochs"] = 1
    training["search_space"]["n_estimators"] = {"low": 10, "high": 30, "type": "int"}
    cfg["evaluation"]["bootstrap_resamples"] = 20
    cfg["explain"]["global_sample_size"] = 300
    yield tmp_path
    cfg.clear()
    cfg.update(saved)


def test_train_then_evaluate_write_a_consistent_servable_bundle(tiny_pipeline):
    train.main()
    evaluate.main()

    models = tiny_pipeline / "models"
    meta = json.loads((models / "metadata.json").read_text(encoding="utf-8"))
    threshold = json.loads((models / "threshold.json").read_text(encoding="utf-8"))
    assert len(meta["models"]) == 9  # M0-M7 plus the tuned XGBoost
    assert all((models / f"{name}.joblib").exists() for name in meta["models"])
    assert meta["best_model"] in load_config()["training"]["deployable_models"]
    assert threshold["model_version"] == meta["model_version"]  # threshold belongs to this model

    reports = tiny_pipeline / "reports"
    assert "## Model comparison (test set)" in (reports / "results.md").read_text(encoding="utf-8")
    summary = json.loads((reports / "test_results.json").read_text(encoding="utf-8"))
    assert len(summary["models"]) == len(meta["models"]) + 1  # plus the calibrated deployed model
    assert all((reports / "figures" / f).stat().st_size > 0 for f in FIGURES)
    assert (reports / "figures" / "shap_summary.png").exists()
    assert (reports / "shap_examples.md").exists()

    with TestClient(create_app()) as client:  # the API serves what was just written
        r = client.post("/predict", json={name: 0.0 for name in FEATURES})
        assert r.status_code == 200
        assert r.json()["model_version"] == meta["model_version"]
        assert r.json()["threshold"] == threshold["threshold"]
