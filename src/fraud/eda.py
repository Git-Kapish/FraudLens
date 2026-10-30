"""EDA plots. Each function saves a figure under reports/figures and returns key numbers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from fraud.config import resolve_path  # noqa: E402
from fraud.data import FEATURES, TARGET  # noqa: E402
from fraud.features import SECONDS_PER_HOUR, hour_of_day  # noqa: E402

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, name: str, out_dir: Path | None) -> Path:
    out_dir = out_dir or resolve_path("figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    logger.info("Saved %s", path)
    return path


def plot_class_balance(df: pd.DataFrame, out_dir: Path | None = None) -> dict[str, float]:
    """Bar chart of class counts (log scale). Returns counts and fraud rate."""
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["legit", "fraud"], counts.values, color=["#4c72b0", "#c44e52"])
    ax.set_yscale("log")
    ax.set_ylabel("transactions (log scale)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    _save(fig, "class_balance.png", out_dir)
    plt.close(fig)
    return {
        "legit": int(counts[0]),
        "fraud": int(counts[1]),
        "fraud_rate": counts[1] / counts.sum(),
    }


def plot_amount_time(df: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """Amount (log1p) and Time histograms by class. Returns Amount summary by class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for label, name, color in [(0, "legit", "#4c72b0"), (1, "fraud", "#c44e52")]:
        sub = df[df[TARGET] == label]
        axes[0].hist(
            np.log1p(sub["Amount"]), bins=50, density=True, alpha=0.6, label=name, color=color
        )
        axes[1].hist(
            sub["Time"] / SECONDS_PER_HOUR,
            bins=48,
            density=True,
            alpha=0.6,
            label=name,
            color=color,
        )
    axes[0].set_xlabel("log1p(Amount)")
    axes[1].set_xlabel("hours since first transaction")
    for ax in axes:
        ax.set_ylabel("density")
        ax.legend()
    _save(fig, "amount_time_by_class.png", out_dir)
    plt.close(fig)
    return df.groupby(TARGET)["Amount"].describe()


def top_separating_features(df: pd.DataFrame, k: int = 8) -> list[str]:
    """Features with the largest absolute correlation with the target."""
    corr = df[FEATURES].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    return list(corr.index[:k])


def plot_top_features(df: pd.DataFrame, k: int = 8, out_dir: Path | None = None) -> list[str]:
    """Boxplots of the k features most correlated with the target, split by class."""
    top = top_separating_features(df, k)
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(3 * ((k + 1) // 2), 6))
    for ax, col in zip(axes.ravel(), top, strict=False):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, showfliers=False, hue=TARGET, legend=False)
        ax.set_xlabel("Class")
    fig.tight_layout()
    _save(fig, "top_features_by_class.png", out_dir)
    plt.close(fig)
    return top


def plot_correlation(df: pd.DataFrame, out_dir: Path | None = None) -> pd.Series:
    """Correlation heatmap of the 30 features plus Class. Returns |corr| with Class."""
    corr = df[[*FEATURES, TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=6)
    _save(fig, "correlation_heatmap.png", out_dir)
    plt.close(fig)
    return corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)


def plot_fraud_by_hour(df: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """Fraud rate and volume by hour (relative to the first transaction)."""
    hours = hour_of_day(df["Time"])
    by_hour = df.groupby(hours)[TARGET].agg(rate="mean", frauds="sum", n="count")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(by_hour.index, by_hour["rate"] * 100, color="#c44e52")
    ax.set_xlabel("hour since first transaction (mod 24)")
    ax.set_ylabel("fraud rate (%)")
    _save(fig, "fraud_rate_by_hour.png", out_dir)
    plt.close(fig)
    return by_hour
