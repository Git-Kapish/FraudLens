import numpy as np
import pytest
from conftest import make_frame
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from fraud.features import FeaturePipeline
from fraud.models import MODEL_NAMES, build_model, build_xgb, scale_pos_weight, score

Y = np.array([0] * 98 + [1] * 2)


def test_scale_pos_weight_hand_computed():
    assert scale_pos_weight(np.array([0] * 8 + [1] * 2)) == 4.0
    assert scale_pos_weight(Y) == 49.0


def test_each_model_uses_its_documented_imbalance_handling():
    assert build_model("M0_dummy", Y).strategy == "prior"
    assert build_model("M1_logreg", Y).class_weight == "balanced"
    assert build_model("M2_rf", Y).class_weight == "balanced_subsample"
    assert build_model("M3_xgb", Y).get_params()["scale_pos_weight"] == 49.0


def test_smote_variant_resamples_inside_the_pipeline_only():
    m4 = build_model("M4_xgb_smote", Y)
    assert isinstance(m4, Pipeline)
    assert [name for name, _ in m4.steps] == ["smote", "clf"]
    assert isinstance(m4.named_steps["smote"], SMOTE)
    # class weighting would double count the imbalance on top of SMOTE
    assert m4.named_steps["clf"].get_params()["scale_pos_weight"] is None


def test_smote_changes_training_only_not_prediction_size():
    # fraud must be rarer than the SMOTE target ratio (0.1), or there is nothing to oversample
    frame = make_frame(n=2500, fraud_rate=0.04).sort_values("Time")
    X, y = FeaturePipeline().fit(frame).transform(frame), frame["Class"].to_numpy()
    model = build_model("M4_xgb_smote", y).fit(X, y)
    assert len(model.predict_proba(X.iloc[:37])) == 37  # SMOTE never runs on the rows we score


def test_all_declared_models_can_be_built_and_unknown_is_rejected():
    for name in MODEL_NAMES:
        assert build_model(name, Y) is not None
    with pytest.raises(ValueError, match="Unknown"):
        build_model("M99", Y)


def test_build_xgb_overrides_defaults():
    assert build_xgb({"max_depth": 2}, spw=3.0).get_params()["max_depth"] == 2
    assert build_xgb(spw=3.0).get_params()["scale_pos_weight"] == 3.0


def test_score_uses_raw_score_when_present_else_probability():
    class Scorer:
        def raw_score(self, X):
            return np.array([0.1, 0.9])

    assert score(Scorer(), None).tolist() == [0.1, 0.9]
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    clf = LogisticRegression().fit(X, [0, 0, 1, 1])
    assert np.allclose(score(clf, X), clf.predict_proba(X)[:, 1])
    assert isinstance(DummyClassifier(strategy="prior"), DummyClassifier)
