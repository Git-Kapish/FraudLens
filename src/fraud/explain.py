"""SHAP explanations for the deployed tree model.

Contributions are in log-odds of the raw XGBoost score, before isotonic calibration, so they
explain the ranking and are not calibrated probabilities. V1-V28 are anonymised PCA
components: explanations name components, they carry no business meaning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from fraud.calibrate import CalibratedModel  # noqa: E402
from fraud.config import load_config, resolve_path, seed_everything  # noqa: E402
from fraud.data import TARGET, load_splits  # noqa: E402
from fraud.models import score  # noqa: E402

logger = logging.getLogger(__name__)


def base_estimator(model: Any) -> Any:
    """The tree model inside a CalibratedModel (or the model itself)."""
    return model.base if isinstance(model, CalibratedModel) else model


def build_explainer(model: Any) -> shap.TreeExplainer:
    """TreeExplainer for the deployed model. Build once and reuse (API does it at startup)."""
    return shap.TreeExplainer(base_estimator(model))


def shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_features) log-odds contributions."""
    return np.asarray(explainer.shap_values(X))


def top_k_contributions(
    explainer: shap.TreeExplainer, X_row: pd.DataFrame, k: int | None = None
) -> list[dict[str, Any]]:
    """Top-k features by absolute contribution for one transaction, with sign.

    `X_row` is a one-row model matrix (the output of FeaturePipeline.transform), so
    `value` is the model-space value (amount_log and hour are standardised).
    """
    k = k or load_config()["explain"]["top_k"]
    contrib = shap_values(explainer, X_row)[0]
    order = np.argsort(-np.abs(contrib))[:k]
    return [
        {
            "feature": X_row.columns[i],
            "value": float(X_row.iloc[0, i]),
            "contribution": float(contrib[i]),
        }
        for i in order
    ]


def _save(fig_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / fig_name, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved %s", out_dir / fig_name)


def global_plots(
    explainer: shap.TreeExplainer, X: pd.DataFrame, y: np.ndarray, out_dir: Path
) -> pd.Series:
    """Beeswarm summary and bar plot on a sample (all frauds + random legit rows).

    Returns mean |SHAP| per feature, sorted descending.
    """
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    fraud_idx = np.flatnonzero(y == 1)
    legit_idx = np.flatnonzero(y == 0)
    n_legit = min(max(cfg["explain"]["global_sample_size"] - len(fraud_idx), 0), len(legit_idx))
    idx = np.sort(np.concatenate([fraud_idx, rng.choice(legit_idx, n_legit, replace=False)]))
    sample = X.iloc[idx]
    values = shap_values(explainer, sample)

    shap.summary_plot(values, sample, show=False, max_display=15)
    _save("shap_summary.png", out_dir)
    shap.summary_plot(values, sample, plot_type="bar", show=False, max_display=15)
    _save("shap_bar.png", out_dir)
    return pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)


def pick_examples(scores: np.ndarray, flagged: np.ndarray, y: np.ndarray, n: int) -> list[int]:
    """Positions of n flagged true frauds spread from highest to lowest score.

    Picked by score only (never by how the explanation looks), to avoid cherry-picking.
    """
    hits = np.flatnonzero(flagged & (y == 1))
    hits = hits[np.argsort(-scores[hits])]
    if len(hits) <= n:
        return list(hits)
    return [int(hits[i]) for i in np.linspace(0, len(hits) - 1, n).round().astype(int)]


def waterfall_plots(
    explainer: shap.TreeExplainer, X: pd.DataFrame, rows: list[int], out_dir: Path
) -> list[Path]:
    """One waterfall PNG per chosen row."""
    paths = []
    values = shap_values(explainer, X.iloc[rows])
    for n, (pos, vals) in enumerate(zip(rows, values, strict=True), start=1):
        explanation = shap.Explanation(
            values=vals,
            base_values=float(explainer.expected_value),
            data=X.iloc[pos].to_numpy(),
            feature_names=list(X.columns),
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        name = f"shap_waterfall_{n}.png"
        _save(name, out_dir)
        paths.append(out_dir / name)
    return paths


def write_examples_md(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    rows: list[int],
    scores: np.ndarray,
    amounts: np.ndarray,
    path: Path,
) -> None:
    """Top-5 signed contributions for each example, as a markdown file."""
    lines = [
        "# SHAP examples (flagged true frauds on the test set)",
        "",
        "Contributions are log-odds of the raw XGBoost score (before calibration). "
        "`V1`-`V28` are anonymised PCA components, so no business meaning is implied. "
        "`amount_log` and `hour` are standardised model-space values. Examples are picked by "
        "score only: highest, middle and lowest-scoring flagged frauds.",
        "",
    ]
    for n, pos in enumerate(rows, start=1):
        top = top_k_contributions(explainer, X.iloc[[pos]])
        lines += [
            f"## Example {n}: calibrated score {scores[pos]:.4f}, Amount {amounts[pos]:,.2f}",
            "",
            "| Feature | Value | Contribution |",
            "|---|---|---|",
            *[f"| {t['feature']} | {t['value']:.3f} | {t['contribution']:+.3f} |" for t in top],
            "",
            f"![waterfall {n}](figures/shap_waterfall_{n}.png)",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    """Global plots on validation, waterfalls and examples on flagged test frauds."""
    seed_everything()
    models_dir = resolve_path("models_dir")
    out_dir = resolve_path("figures_dir")
    pipeline = joblib.load(models_dir / "pipeline.joblib")  # trusted: written by train.py
    deployed = joblib.load(models_dir / "best_calibrated.joblib")
    threshold = json.loads((models_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    explainer = build_explainer(deployed)
    splits = load_splits()

    X_val = pipeline.transform(splits["val"])
    importance = global_plots(explainer, X_val, splits["val"][TARGET].to_numpy(), out_dir)
    logger.info("Top features by mean |SHAP|: %s", importance.head(5).round(3).to_dict())

    X_test = pipeline.transform(splits["test"])
    y_test = splits["test"][TARGET].to_numpy()
    s_test = score(deployed, X_test)
    rows = pick_examples(
        s_test, s_test >= threshold, y_test, load_config()["explain"]["n_waterfalls"]
    )
    waterfall_plots(explainer, X_test, rows, out_dir)
    write_examples_md(
        explainer,
        X_test,
        rows,
        s_test,
        splits["test"]["Amount"].to_numpy(),
        resolve_path("results_md").parent / "shap_examples.md",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()


if __name__ == "__main__":
    main()
