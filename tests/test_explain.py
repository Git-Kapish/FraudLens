import numpy as np
import pandas as pd
import pytest
from conftest import make_frame
from xgboost import XGBClassifier

from fraud.calibrate import CalibratedModel, fit_isotonic
from fraud.explain import (
    base_estimator,
    build_explainer,
    global_plots,
    pick_examples,
    top_k_contributions,
    waterfall_plots,
)
from fraud.features import FeaturePipeline
from fraud.models import build_model


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 8)), columns=[f"f{i}" for i in range(8)])
    y = (X["f0"] + 0.5 * X["f3"] + rng.normal(scale=0.3, size=500) > 1).astype(int).to_numpy()
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0).fit(X, y)
    return model, X, y


def test_top_k_is_sorted_signed_and_has_k_entries(fitted):
    model, X, _ = fitted
    top = top_k_contributions(build_explainer(model), X.iloc[[0]], k=5)
    assert len(top) == 5
    mags = [abs(t["contribution"]) for t in top]
    assert mags == sorted(mags, reverse=True)
    assert {"feature", "value", "contribution"} == set(top[0])


def test_contributions_add_up_to_the_model_margin(fitted):
    model, X, _ = fitted
    explainer = build_explainer(model)
    row = X.iloc[[3]]
    all_features = top_k_contributions(explainer, row, k=X.shape[1])
    total = sum(t["contribution"] for t in all_features) + float(explainer.expected_value)
    assert total == pytest.approx(float(model.predict(row, output_margin=True)[0]), abs=1e-4)


def test_explainer_unwraps_calibrated_model(fitted):
    model, X, y = fitted
    raw = model.predict_proba(X)[:, 1]
    calibrated = CalibratedModel(model, fit_isotonic(raw, y))
    assert base_estimator(calibrated) is model
    assert build_explainer(calibrated) is not None


def test_global_and_waterfall_plots_write_files(fitted, tmp_path):
    model, X, y = fitted
    explainer = build_explainer(model)
    # fewer legit rows than global_sample_size: must sample what exists, not crash
    importance = global_plots(explainer, X, y, tmp_path)
    assert importance.is_monotonic_decreasing and set(importance.index) == set(X.columns)
    assert (tmp_path / "shap_summary.png").stat().st_size > 0
    assert (tmp_path / "shap_bar.png").stat().st_size > 0
    paths = waterfall_plots(explainer, X, [0, 5], tmp_path)
    assert [p.name for p in paths] == ["shap_waterfall_1.png", "shap_waterfall_2.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_smote_pipeline_model_can_be_explained():
    # M4 is deployable (config.yaml); TreeExplainer cannot take an imblearn Pipeline directly
    frame = make_frame(n=2500, fraud_rate=0.04).sort_values("Time")
    X, y = FeaturePipeline().fit(frame).transform(frame), frame["Class"].to_numpy()
    m4 = build_model("M4_xgb_smote", y).fit(X, y)
    calibrated = CalibratedModel(m4, fit_isotonic(m4.predict_proba(X)[:, 1], y))
    xgb = m4.named_steps["clf"]
    assert base_estimator(calibrated) is xgb
    explainer = build_explainer(calibrated)
    row = X.iloc[[0]]
    total = sum(t["contribution"] for t in top_k_contributions(explainer, row, k=X.shape[1]))
    margin = float(xgb.predict(row, output_margin=True)[0])
    assert total + float(explainer.expected_value) == pytest.approx(margin, abs=1e-4)


def test_waterfall_plots_with_no_flagged_frauds_is_a_no_op(fitted, tmp_path):
    model, X, _ = fitted
    # SHAP raises IndexError on an empty frame; a run that flags no fraud must not crash
    assert waterfall_plots(build_explainer(model), X, [], tmp_path) == []


def test_pick_examples_spreads_over_flagged_frauds_only():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    y = np.array([1, 0, 1, 1, 1, 0, 1])
    flagged = scores >= 0.4  # row 6 (fraud, 0.3) is not flagged
    rows = pick_examples(scores, flagged, y, n=3)
    assert rows == [0, 3, 4]  # frauds flagged are rows 0, 2, 3, 4: highest, middle, lowest
    assert all(flagged[r] and y[r] == 1 for r in rows)
