import numpy as np
import pytest

from fraud.data import FEATURES, time_split
from fraud.features import DERIVED_COLUMNS, MODEL_COLUMNS, FeaturePipeline, _derive


def test_scaler_stats_equal_train_stats(frame):
    parts = time_split(frame)
    pipe = FeaturePipeline().fit(parts["train"])
    train_derived = _derive(parts["train"])
    assert np.allclose(pipe.scaler.mean_, train_derived.mean().to_numpy())
    assert np.allclose(pipe.scaler.scale_, train_derived.std(ddof=0).to_numpy())


def test_transform_does_not_refit_on_val_or_test(frame):
    parts = time_split(frame)
    pipe = FeaturePipeline().fit(parts["train"])
    mean_before, scale_before = pipe.scaler.mean_.copy(), pipe.scaler.scale_.copy()
    val_out = pipe.transform(parts["val"])
    pipe.transform(parts["test"])
    assert np.array_equal(pipe.scaler.mean_, mean_before)
    assert np.array_equal(pipe.scaler.scale_, scale_before)
    # train-fitted scaling leaves val unstandardised (hour drifts over time)
    assert not np.isclose(val_out["hour"].mean(), 0.0, atol=1e-6)


def test_output_columns_and_no_raw_time(frame):
    out = FeaturePipeline().fit(frame).transform(frame)
    assert list(out.columns) == MODEL_COLUMNS
    assert "Time" not in out.columns
    assert set(DERIVED_COLUMNS) <= set(out.columns)


def test_missing_feature_raises(frame):
    pipe = FeaturePipeline().fit(frame)
    with pytest.raises(ValueError, match="Missing"):
        pipe.transform(frame.drop(columns=[FEATURES[1]]))
