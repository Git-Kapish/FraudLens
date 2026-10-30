"""Test-set evaluation. Run once, after training has frozen every choice.

Thresholds are chosen on validation by minimising cost, then applied unchanged to test.
Accuracy is deliberately not computed: with 0.17% fraud it says nothing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from fraud.config import load_config, resolve_path, seed_everything
from fraud.cost import best_threshold, cost_by_threshold, flag_nothing_cost, savings, total_cost
from fraud.data import TARGET, load_raw, prepare
from fraud.models import score

logger = logging.getLogger(__name__)
DEPLOYED = "best_calibrated"
F2_BETA = 2.0
DEFAULT_THRESHOLD = 0.5


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    """Confusion counts for a boolean decision."""
    y, flagged = np.asarray(y) == 1, np.asarray(flagged, dtype=bool)
    return {
        "tp": int((y & flagged).sum()),
        "fp": int((~y & flagged).sum()),
        "fn": int((y & ~flagged).sum()),
        "tn": int((~y & ~flagged).sum()),
    }


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta score; 0 when precision and recall are both 0."""
    denom = beta**2 * precision + recall
    return (1 + beta**2) * precision * recall / denom if denom else 0.0


def operating_point(y: np.ndarray, flagged: np.ndarray) -> dict[str, Any]:
    """Precision, recall, F1, F2 and confusion counts at a fixed decision."""
    c = confusion(y, flagged)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f_beta(precision, recall, 1.0),
        "f2": f_beta(precision, recall, F2_BETA),
        "n_flagged": c["tp"] + c["fp"],
        **c,
    }


def recall_at_precision(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Highest recall reachable with precision >= target (0 if never reached)."""
    p, r, _ = precision_recall_curve(y, s)
    return float(r[p >= target].max()) if (p >= target).any() else 0.0


def precision_at_recall(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Highest precision reachable with recall >= target (0 if never reached)."""
    p, r, _ = precision_recall_curve(y, s)
    return float(p[r >= target].max()) if (r >= target).any() else 0.0


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_resamples: int,
    level: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI for each metric, using the same resamples for all of them."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    while len(draws[next(iter(draws))]) < n_resamples:
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:  # a resample with no frauds has no PR curve
            continue
        for name, fn in metrics.items():
            draws[name].append(fn(y[idx], s[idx]))
    alpha = (1 - level) / 2
    return {k: tuple(np.quantile(v, [alpha, 1 - alpha])) for k, v in draws.items()}


def load_artifacts() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Return (metadata, feature pipeline, {name: fitted model incl. deployed}).

    joblib is pickle: only load files written by train.py in this repo's models/ dir.
    """
    models_dir = resolve_path("models_dir")
    meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    pipeline = joblib.load(models_dir / "pipeline.joblib")
    models = {n: joblib.load(models_dir / f"{n}.joblib") for n in meta["models"]}
    models[DEPLOYED] = joblib.load(models_dir / f"{DEPLOYED}.joblib")
    return meta, pipeline, models


def pick_ci_models(meta: dict[str, Any]) -> list[str]:
    """Top 2 by validation PR-AUC (a frozen choice, not made on test) plus the deployed model."""
    ranked = sorted(meta["val_pr_auc"], key=meta["val_pr_auc"].get, reverse=True)
    return [*ranked[:2], DEPLOYED]


def evaluate_models(
    models: dict[str, Any],
    X: dict[str, pd.DataFrame],
    y: dict[str, np.ndarray],
    amount: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Score every model on val and test once and compute all test metrics."""
    cfg = load_config()
    rc = cfg["cost"]["review_cost"]
    ev = cfg["evaluation"]
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        s_val, s_test = score(model, X["val"]), score(model, X["test"])
        thr, val_cost = best_threshold(y["val"], s_val, amount["val"], rc)
        flagged = s_test >= thr
        test_cost = total_cost(y["test"], flagged, amount["test"], rc)
        base = flag_nothing_cost(y["test"], amount["test"])
        row = {
            "s_val": s_val,
            "s_test": s_test,
            "threshold": thr,
            "val_cost": val_cost,
            "test_pr_auc": average_precision_score(y["test"], s_test),
            "test_roc_auc": roc_auc_score(y["test"], s_test),
            "recall_at_precision": recall_at_precision(y["test"], s_test, ev["precision_target"]),
            "precision_at_recall": precision_at_recall(y["test"], s_test, ev["recall_target"]),
            "test_cost": test_cost,
            "flag_nothing_cost": base,
            **operating_point(y["test"], flagged),
        }
        row["saving"], row["saving_pct"] = savings(test_cost, base)
        if hasattr(model, "predict_proba"):  # 0.5 only means something for probabilities
            cost_half = total_cost(y["test"], s_test >= DEFAULT_THRESHOLD, amount["test"], rc)
            row["cost_at_0.5"] = cost_half
            row["saving_vs_0.5"], row["saving_vs_0.5_pct"] = savings(cost_half, base)
        results[name] = row
        logger.info("%s: test PR-AUC %.4f", name, row["test_pr_auc"])
    return results


def run() -> dict[str, Any]:
    """Full evaluation: score, CIs, cost analysis, sensitivity, plots, report, threshold file."""
    from fraud import report  # imported here so tests of the metrics need no plotting libs

    cfg = load_config()
    seed_everything()
    meta, pipeline, models = load_artifacts()
    raw = load_raw()
    splits = prepare(raw)
    X = {k: pipeline.transform(splits[k]) for k in ("val", "test")}
    y = {k: splits[k][TARGET].to_numpy() for k in ("val", "test")}
    amount = {k: splits[k]["Amount"].to_numpy() for k in ("val", "test")}

    results = evaluate_models(models, X, y, amount)
    ev = cfg["evaluation"]

    metric_fns = {
        "pr_auc": average_precision_score,
        "recall_at_precision": lambda yy, ss: recall_at_precision(yy, ss, ev["precision_target"]),
    }
    cis = {
        name: bootstrap_ci(
            y["test"],
            results[name]["s_test"],
            metric_fns,
            ev["bootstrap_resamples"],
            ev["ci_level"],
            cfg["seed"],
        )
        for name in pick_ci_models(meta)
    }

    deployed = results[DEPLOYED]
    sensitivity = []
    for rc in cfg["cost"]["sensitivity_review_costs"]:
        thr, _ = best_threshold(y["val"], deployed["s_val"], amount["val"], rc)
        flagged = deployed["s_test"] >= thr
        cost = total_cost(y["test"], flagged, amount["test"], rc)
        base = flag_nothing_cost(y["test"], amount["test"])
        saved, pct = savings(cost, base)
        op = operating_point(y["test"], flagged)
        sensitivity.append(
            {
                "review_cost": rc,
                "threshold": thr,
                "test_cost": cost,
                "flag_nothing_cost": base,
                "saving": saved,
                "saving_pct": pct,
                **{k: op[k] for k in ("precision", "recall", "n_flagged")},
            }
        )

    curves = {
        split: cost_by_threshold(
            y[split], deployed[f"s_{split}"], amount[split], cfg["cost"]["review_cost"]
        )
        for split in ("val", "test")
    }
    out = {
        "meta": meta,
        "results": results,
        "cis": cis,
        "sensitivity": pd.DataFrame(sensitivity),
        "curves": curves,
        "y": y,
        "split_sizes": {k: (len(splits[k]), int(splits[k][TARGET].sum())) for k in splits},
        "dedup_removed": int(raw.duplicated().sum()),
    }
    threshold_path = resolve_path("models_dir") / "threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": deployed["threshold"],
                "review_cost": cfg["cost"]["review_cost"],
                "model": DEPLOYED,
                "model_version": meta["model_version"],
                "selected_on": "validation, minimising cost",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report.make_figures(out)
    report.write_results(out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
    logger.info("Wrote %s", resolve_path("results_md"))


if __name__ == "__main__":
    main()
