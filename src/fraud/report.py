"""Figures and reports/results.md from the evaluation output."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from fraud.config import load_config, resolve_path  # noqa: E402
from fraud.models import DISPLAY_NAMES  # noqa: E402

logger = logging.getLogger(__name__)
DEPLOYED = "best_calibrated"
BASELINE_MODEL = "M0_dummy"


def label(name: str, meta: dict[str, Any]) -> str:
    """Human-readable model name; the deployed model names its base."""
    if name == DEPLOYED:
        return f"Deployed: {DISPLAY_NAMES[meta['best_model']]}, calibrated"
    return DISPLAY_NAMES[name]


def _save(fig: plt.Figure, name: str) -> None:
    out = resolve_path("figures_dir")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out / name)


def _overlay(out: dict[str, Any], kind: str) -> None:
    """PR or ROC curves for every model on the test set."""
    y, meta = out["y"]["test"], out["meta"]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (name, r) in enumerate(out["results"].items()):
        if kind == "pr":
            p, rec, _ = precision_recall_curve(y, r["s_test"])
            x, yy, metric = rec, p, r["test_pr_auc"]
        else:
            fpr, tpr, _ = roc_curve(y, r["s_test"])
            x, yy, metric = fpr, tpr, r["test_roc_auc"]
        style = ":" if name == BASELINE_MODEL else "-"
        width = 2.5 if name == DEPLOYED else 1.2
        ax.plot(
            x,
            yy,
            style,
            lw=width,
            color=colors[i % 10],
            label=f"{label(name, meta)} ({metric:.3f})",
        )
    if kind == "pr":
        ax.set(
            xlabel="recall", ylabel="precision", title="Precision-recall, test set (legend: PR-AUC)"
        )
    else:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(
            xlabel="false positive rate",
            ylabel="true positive rate",
            title="ROC, test set (legend: ROC-AUC)",
        )
    ax.legend(fontsize=7, loc="lower left" if kind == "pr" else "lower right")
    _save(fig, "pr_curves.png" if kind == "pr" else "roc_curves.png")


def _cost_curve(out: dict[str, Any]) -> None:
    """Total cost versus threshold for the deployed model, with the chosen threshold marked."""
    rc = load_config()["cost"]["review_cost"]
    thr = out["results"][DEPLOYED]["threshold"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in ("val", "test"):
        c = out["curves"][split]
        c = c[c["threshold"] > 0]  # log axis; a threshold of 0 flags everything
        ax.plot(c["threshold"], c["cost"], label=f"{split} cost")
    ax.axvline(thr, color="k", ls="--", label=f"chosen threshold {thr:.4f} (from val)")
    ax.axhline(
        out["results"][DEPLOYED]["flag_nothing_cost"],
        color="gray",
        ls=":",
        label="flag nothing (test)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlabel="threshold",
        ylabel="total cost (currency units)",
        title=f"Cost vs threshold, review cost = {rc}",
    )
    ax.legend(fontsize=8)
    _save(fig, "cost_vs_threshold.png")


def _confusion(out: dict[str, Any]) -> None:
    r = out["results"][DEPLOYED]
    m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(
            j, i, f"{v:,}", ha="center", va="center", color="white" if v > m.max() / 2 else "black"
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["legit", "fraud"],
        yticklabels=["legit", "fraud"],
        xlabel="predicted",
        ylabel="actual",
        title="Deployed model, test",
    )
    _save(fig, "confusion_matrix.png")


def make_figures(out: dict[str, Any]) -> None:
    """PR/ROC overlays, cost curve and confusion matrix."""
    _overlay(out, "pr")
    _overlay(out, "roc")
    _cost_curve(out)
    _confusion(out)


def md_table(df: pd.DataFrame) -> str:
    """Render a frame as a GitHub markdown table."""
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


LATENCY_START = "<!-- latency:start -->"
LATENCY_END = "<!-- latency:end -->"


def load_latency() -> dict[str, Any] | None:
    """Latest scripts/replay.py measurement, if one was saved."""
    path = resolve_path("results_md").parent / "latency.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def latency_block(data: dict[str, Any] | None) -> str:
    """Markdown for the latency section body."""
    if data is None:
        return (
            "_Not measured yet. Run `make serve`, then `python scripts/replay.py --write-report`._"
        )
    verdict = "met" if data["meets_target"] else "NOT met"
    return (
        f"`POST /predict`, {data['n_calls']} sequential calls on localhost, one worker, "
        f"after a warm-up. Target: p95 < {data['target_p95_ms']} ms ({verdict}).\n\n"
        "| | p50 (ms) | p95 (ms) |\n|---|---|---|\n"
        f"| Server-side `latency_ms` | {data['server_p50_ms']:.2f} | {data['server_p95_ms']:.2f} |\n"
        f"| Client round trip | {data['round_trip_p50_ms']:.2f} | {data['round_trip_p95_ms']:.2f} |"
    )


def update_latency(data: dict[str, Any]) -> None:
    """Replace the latency section of an existing results.md in place."""
    path = resolve_path("results_md")
    text = path.read_text(encoding="utf-8")
    block = f"{LATENCY_START}\n{latency_block(data)}\n{LATENCY_END}"
    pattern = re.compile(f"{re.escape(LATENCY_START)}.*?{re.escape(LATENCY_END)}", re.DOTALL)
    if not pattern.search(text):
        raise ValueError("results.md has no latency section; run `make evaluate` first")
    path.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")


def val_pr_auc(name: str, meta: dict[str, Any]) -> float:
    """Validation PR-AUC from metadata (the deployed model has its own calibrated value)."""
    if name == DEPLOYED:
        return meta["calibrated_val"]["val_pr_auc"]
    return meta["val_pr_auc"][name]


def caveats(out: dict[str, Any]) -> str:
    """Caveat bullets that quote the measured numbers, so they cannot drift from the tables."""
    meta, results = out["meta"], out["results"]
    d, base = results[DEPLOYED], results[meta["best_model"]]
    best = max(results, key=lambda n: results[n]["test_pr_auc"])
    vs_half = "lower" if d["test_cost"] < d["cost_at_0.5"] else "no better"
    target = load_config()["evaluation"]["prd_pr_auc_target"]
    bounds = [out["cis"][n]["pr_auc"] for n in (DEPLOYED, best) if n in out["cis"]]
    spans = bounds and all(lo <= target <= hi for lo, hi in bounds)
    span_note = "Their intervals span the target. " if spans else ""
    return f"""- The test set has {out["split_sizes"]["test"][1]} frauds, so every test number is noisy. Use the intervals, not point estimates, to compare models; the top models are statistically indistinguishable.
- A chronological split is harder than the random splits behind most published numbers on this dataset, because fraud patterns drift and later rows are never seen in training. Test PR-AUC ({d["test_pr_auc"]:.3f} deployed) is below validation ({val_pr_auc(DEPLOYED, meta):.3f}). That drop is expected and is not hidden.
- PRD target of PR-AUC >= {target}: the deployed model scores {d["test_pr_auc"]:.3f}, the best model on test ({label(best, meta)}) scores {results[best]["test_pr_auc"]:.3f}. {span_note}No tuning was done to hit it.
- Deployed model selection was frozen on validation (best XGBoost-family model, chosen for TreeExplainer and latency), so it was not switched after seeing test results, even though {label(best, meta)} scores higher on test within noise.
- Isotonic calibration makes scores piecewise-constant, which creates ties. That lowers the deployed model's PR-AUC ({d["test_pr_auc"]:.3f}) versus its uncalibrated base ({base["test_pr_auc"]:.3f}), and can make nearby review costs pick the same threshold (see sensitivity table).
- The validation-chosen threshold ({d["threshold"]:.4g}) costs {d["test_cost"]:,.0f} on test versus {d["cost_at_0.5"]:,.0f} at 0.5, i.e. {vs_half} than the 0.5 baseline. The difference is a few transactions; the cost model does not show a reliable advantage over 0.5 for this calibrated model.
- Isolation Forest and the autoencoder rank poorly on their own; the ensemble that includes the autoencoder is reported even though it does not beat XGBoost.
- Costs depend on the assumed review cost and on a single two-day window of data."""


def _ci(ci: tuple[float, float], point: float) -> str:
    return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def write_results(out: dict[str, Any]) -> None:
    """Write reports/results.md with every table and figure link."""
    cfg = load_config()
    meta, results = out["meta"], out["results"]
    rc = cfg["cost"]["review_cost"]
    d = results[DEPLOYED]
    order = sorted(results, key=lambda n: results[n]["test_pr_auc"], reverse=True)

    compare = pd.DataFrame(
        [
            {
                "Model": label(n, meta),
                "Val PR-AUC": f"{val_pr_auc(n, meta):.3f}",
                "Test PR-AUC": f"{results[n]['test_pr_auc']:.3f}",
                "Test ROC-AUC": f"{results[n]['test_roc_auc']:.3f}",
                "Recall @ 90% precision": f"{results[n]['recall_at_precision']:.3f}",
                "Precision @ 80% recall": f"{results[n]['precision_at_recall']:.3f}",
            }
            for n in order
        ]
    )
    ci = pd.DataFrame(
        [
            {
                "Model": label(n, meta),
                "Test PR-AUC [95% CI]": _ci(c["pr_auc"], results[n]["test_pr_auc"]),
                "Recall @ 90% precision [95% CI]": _ci(
                    c["recall_at_precision"], results[n]["recall_at_precision"]
                ),
            }
            for n, c in out["cis"].items()
        ]
    )
    op = pd.DataFrame(
        [
            {
                "Model": label(n, meta),
                "Threshold (val)": f"{results[n]['threshold']:.4g}",
                "Precision": f"{results[n]['precision']:.3f}",
                "Recall": f"{results[n]['recall']:.3f}",
                "F1": f"{results[n]['f1']:.3f}",
                "F2": f"{results[n]['f2']:.3f}",
                "Flagged": results[n]["n_flagged"],
                "Test cost": f"{results[n]['test_cost']:,.0f}",
                "Saving vs flag-nothing": f"{results[n]['saving_pct']:.1f}%",
            }
            for n in order
        ]
    )
    sens = out["sensitivity"].copy()
    sens = pd.DataFrame(
        {
            "Review cost": sens["review_cost"],
            "Threshold (val)": sens["threshold"].map("{:.4g}".format),
            "Flagged": sens["n_flagged"],
            "Precision": sens["precision"].map("{:.3f}".format),
            "Recall": sens["recall"].map("{:.3f}".format),
            "Test cost": sens["test_cost"].map("{:,.0f}".format),
            "Flag-nothing cost": sens["flag_nothing_cost"].map("{:,.0f}".format),
            "Saving": sens["saving"].map("{:,.0f}".format),
            "Saving %": sens["saving_pct"].map("{:.1f}%".format),
        }
    )
    sizes = pd.DataFrame(
        [{"Split": k, "Rows": f"{n:,}", "Frauds": f} for k, (n, f) in out["split_sizes"].items()]
    )
    half = (
        f"| Threshold 0.5 | {d['cost_at_0.5']:,.0f} | {d['saving_vs_0.5']:,.0f} ({d['saving_vs_0.5_pct']:.1f}%) |\n"
        if "cost_at_0.5" in d
        else ""
    )

    text = f"""# Results

Generated by `make evaluate` (`src/fraud/evaluate.py`). Model version `{meta["model_version"]}`, seed {meta["seed"]}.
The test set was scored only after training, tuning, calibration and threshold choice were frozen on train/validation.
Headline metric is **PR-AUC (average precision)**. Accuracy is not reported: with 0.17% fraud it carries no information.
Amounts are in the dataset's currency units (the dataset does not state the currency).

## Data
Exact duplicates removed before splitting: **{out["dedup_removed"]:,}**. Chronological split by `Time` (70/15/15):

{md_table(sizes)}

## Model comparison (test set)
Sorted by test PR-AUC. ROC-AUC is secondary and optimistic here: with so few frauds the huge legit class keeps the false positive rate tiny, so ROC-AUC looks high even for a poor ranker.

{md_table(compare)}

## Bootstrap 95% confidence intervals
{cfg["evaluation"]["bootstrap_resamples"]} bootstrap resamples of the test set. Models shown: the top 2 by *validation* PR-AUC (chosen before looking at test) plus the deployed model. With only {out["split_sizes"]["test"][1]} test frauds the intervals are wide; differences between close models are not meaningful.

{md_table(ci)}

## Operating point at the cost-optimal threshold
Each model's threshold is chosen on **validation** by minimising total cost (review cost = {rc}), then applied unchanged to test.

{md_table(op)}

## Cost analysis (deployed model)
**`review_cost = {rc}` is an assumption, not a measured value.** Cost model: a missed fraud costs its `Amount`; every flagged transaction (true or false positive) costs `review_cost`; a true negative costs 0.

| Strategy | Test cost | Saving vs flag-nothing |
|---|---|---|
| Flag nothing | {d["flag_nothing_cost"]:,.0f} | - |
| Deployed model at val-chosen threshold {d["threshold"]:.4g} | {d["test_cost"]:,.0f} | {d["saving"]:,.0f} ({d["saving_pct"]:.1f}%) |
{half}
Confusion matrix on test at the chosen threshold: TP {d["tp"]}, FP {d["fp"]}, FN {d["fn"]}, TN {d["tn"]:,}.

![Cost vs threshold](figures/cost_vs_threshold.png)
![Confusion matrix](figures/confusion_matrix.png)

### Sensitivity to review cost
The threshold is re-chosen on validation for each review cost, then applied to test.

{md_table(sens)}

## Curves
![PR curves](figures/pr_curves.png)
![ROC curves](figures/roc_curves.png)
![Reliability curve](figures/reliability_curve.png)

## API latency
{LATENCY_START}
{latency_block(load_latency())}
{LATENCY_END}

## Caveats
{caveats(out)}
"""
    path = resolve_path("results_md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
