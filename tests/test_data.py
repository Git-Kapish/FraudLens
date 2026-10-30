import pandas as pd
import pytest

from fraud.config import load_config
from fraud.data import FEATURES, TARGET, dedupe, prepare, time_split, validate_schema


def test_split_sizes(frame):
    parts = time_split(frame)
    n = len(frame)
    assert sum(len(p) for p in parts.values()) == n
    assert len(parts["train"]) == pytest.approx(0.70 * n, abs=1)
    assert len(parts["val"]) == pytest.approx(0.15 * n, abs=2)
    assert len(parts["test"]) == pytest.approx(0.15 * n, abs=2)


def test_split_is_chronological_without_overlap(frame):
    parts = time_split(frame)
    assert parts["train"]["Time"].is_monotonic_increasing
    assert parts["train"]["Time"].max() <= parts["val"]["Time"].min()
    assert parts["val"]["Time"].max() <= parts["test"]["Time"].min()


def test_dedup_happens_before_split(frame):
    dup = pd.concat([frame, frame.iloc[:200]], ignore_index=True)
    parts = prepare(dup)
    assert sum(len(p) for p in parts.values()) == len(frame)
    combined = pd.concat(parts.values())
    assert not combined.duplicated().any()


def test_dedupe_reports_removed_count(frame):
    out, removed = dedupe(pd.concat([frame, frame.iloc[:10]], ignore_index=True))
    assert removed == 10
    assert len(out) == len(frame)


def test_too_few_frauds_raises(frame):
    frame[TARGET] = 0
    frame.loc[frame.index[:5], TARGET] = 1
    with pytest.raises(ValueError, match="frauds"):
        time_split(frame, load_config())


def test_validate_schema_flags_missing_column(frame):
    with pytest.raises(ValueError, match="Missing"):
        validate_schema(frame.drop(columns=["V3"]))


def test_validate_schema_report(frame):
    report = validate_schema(frame)
    assert report["rows"] == len(frame)
    assert report["nulls"] == 0
    assert len(FEATURES) == 30
