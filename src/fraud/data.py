"""Load, validate, deduplicate and chronologically split the ULB dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fraud.config import load_config, resolve_path

logger = logging.getLogger(__name__)

TARGET = "Class"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
SPLIT_NAMES = ("train", "val", "test")


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw Kaggle CSV."""
    path = path or resolve_path("raw_data")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` (see scripts/download_data.py)."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Check required columns and return a report of nulls, dtypes and duplicates."""
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError("Class must be binary (0/1)")
    report = {
        "rows": len(df),
        "nulls": int(df[[*FEATURES, TARGET]].isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "non_numeric": [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])],
        "frauds": int(df[TARGET].sum()),
    }
    if report["nulls"] or report["non_numeric"]:
        raise ValueError(f"Bad data: {report}")
    logger.info("Schema report: %s", report)
    return report


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows. Returns the frame and how many rows were removed."""
    out = df.drop_duplicates().reset_index(drop=True)
    removed = len(df) - len(out)
    logger.info("Removed %d duplicate rows (%d -> %d)", removed, len(df), len(out))
    return out, removed


def time_split(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Chronological train/val/test split by `Time`.

    Rows are sorted by Time and cut by position, so a Time value shared across a
    boundary can appear on both sides (Time has 1-second resolution).
    """
    split_cfg = (cfg or load_config())["split"]
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    i1 = int(n * split_cfg["train"])
    i2 = int(n * (split_cfg["train"] + split_cfg["val"]))
    parts = dict(zip(SPLIT_NAMES, [df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]], strict=True))

    for name, part in parts.items():
        frauds = int(part[TARGET].sum())
        logger.info("%s: %d rows, %d frauds", name, len(part), frauds)
        minimum = split_cfg["min_frauds_per_split"]
        if frauds < minimum:
            raise ValueError(f"{name} split has {frauds} frauds, need at least {minimum}")
    return parts


def prepare(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Validate, deduplicate (before splitting) and split."""
    cfg = cfg or load_config()
    validate_schema(df)
    if cfg["split"]["dedup"]:
        df, _ = dedupe(df)
    return time_split(df, cfg)


def load_splits() -> dict[str, pd.DataFrame]:
    """Load the raw CSV and return the three chronological splits."""
    return prepare(load_raw())


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the 30 feature columns and the target."""
    return df[FEATURES], df[TARGET]
