"""Streamlit analyst dashboard. Layout only: all logic lives in src/fraud.

Needs data/raw/creditcard.csv (`make data`) and the artifacts from `make train evaluate`.
The Live replay page also needs the API (`make serve`).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud.api.main import load_artifacts
from fraud.config import load_config, resolve_path
from fraud.cost import best_threshold, cost_by_threshold, flag_nothing_cost, savings, total_cost
from fraud.data import TARGET, dedupe, load_raw, time_split, validate_schema
from fraud.evaluate import DEFAULT_THRESHOLD, operating_point
from fraud.explain import top_k_contributions
from fraud.features import SECONDS_PER_HOUR
from fraud.replay import latency_summary, replay

CFG = load_config()
PAGES = [
    "Overview",
    "Model comparison",
    "Threshold and cost",
    "Transaction explorer",
    "Live replay",
]
FEED_ROWS = 15
UPDATE_EVERY = 5  # redraw the replay page every N transactions
st.set_page_config(page_title="FraudLens", layout="wide")


# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading data...")
def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], int]:
    """Deduplicated data, its chronological splits and the number of duplicates removed."""
    try:
        raw = load_raw()
    except FileNotFoundError:
        st.error("data/raw/creditcard.csv not found. Run `make data` first.")
        st.stop()
    validate_schema(raw)
    df, removed = dedupe(raw)
    return df, time_split(df), removed


@st.cache_resource(show_spinner="Loading model...")
def load_bundle() -> Any:
    """Pipeline, calibrated model, SHAP explainer and threshold (same loader as the API)."""
    try:
        return load_artifacts()
    except FileNotFoundError:
        st.error("Model files not found in models/. Run `make train evaluate` first.")
        st.stop()


@st.cache_resource(show_spinner="Scoring...")
def score_split(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Model matrix, calibrated scores and raw (pre-calibration) scores for val or test."""
    art, (_, splits, _) = load_bundle(), load_data()
    X = art.pipeline.transform(splits[split])
    return X, art.model.predict_proba(X)[:, 1], art.model.base.predict_proba(X)[:, 1]


def load_results() -> dict[str, Any]:
    path = resolve_path("results_md").parent / "test_results.json"
    if not path.exists():
        st.error("reports/test_results.json not found. Run `make evaluate` first.")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- pages
def page_overview() -> None:
    df, splits, removed = load_data()
    n, frauds = len(df), int(df[TARGET].sum())
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{n:,}", help="After removing exact duplicate rows")
    b.metric("Frauds", f"{frauds:,}")
    c.metric("Fraud rate", f"{frauds / n:.3%}")
    d.metric("Duplicates removed", f"{removed:,}")

    left, right = st.columns([1, 2])
    counts = df[TARGET].map({0: "legit", 1: "fraud"}).value_counts()
    fig = px.bar(x=counts.index, y=counts.values, log_y=True, labels={"x": "", "y": "rows (log)"})
    fig.update_layout(title="Class balance", height=350)
    left.plotly_chart(fig, width="stretch")

    hourly = df.groupby((df["Time"] // SECONDS_PER_HOUR).astype(int))[TARGET].agg(
        rate="mean", frauds="sum", rows="count"
    )
    fig = px.bar(
        hourly.reset_index(),
        x="Time",
        y="rate",
        hover_data=["frauds", "rows"],
        labels={"Time": "hours since first transaction", "rate": "fraud rate"},
    )
    for name in ("val", "test"):  # mark where each split starts
        fig.add_vline(
            x=splits[name]["Time"].min() / SECONDS_PER_HOUR, line_dash="dash", annotation_text=name
        )
    fig.update_layout(title="Fraud rate over time", height=350, yaxis_tickformat=".2%")
    right.plotly_chart(fig, width="stretch")

    table = pd.DataFrame(
        {
            name: {
                "rows": len(s),
                "frauds": int(s[TARGET].sum()),
                "fraud rate": f"{s[TARGET].mean():.3%}",
                "from hour": round(s["Time"].min() / SECONDS_PER_HOUR, 1),
                "to hour": round(s["Time"].max() / SECONDS_PER_HOUR, 1),
            }
            for name, s in splits.items()
        }
    ).T
    st.subheader("Chronological split (70/15/15 by Time)")
    st.dataframe(table, width="stretch")
    st.caption(
        "V1-V28 are anonymised PCA components and carry no business meaning. "
        "The fraud rate by hour is relative to the first transaction (clock time is unknown)."
    )


def page_models() -> None:
    res = load_results()
    models = res["models"]
    order = sorted(models, key=lambda m: models[m]["test_pr_auc"], reverse=True)
    st.subheader("Test-set comparison (headline: PR-AUC)")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": [models[m]["label"] for m in order],
                "Val PR-AUC": [models[m]["val_pr_auc"] for m in order],
                "Test PR-AUC": [models[m]["test_pr_auc"] for m in order],
                "Test ROC-AUC": [models[m]["test_roc_auc"] for m in order],
                "Recall @ 90% precision": [models[m]["recall_at_precision"] for m in order],
                "Precision @ 80% recall": [models[m]["precision_at_recall"] for m in order],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in (
                "Val PR-AUC",
                "Test PR-AUC",
                "Test ROC-AUC",
                "Recall @ 90% precision",
                "Precision @ 80% recall",
            )
        },
    )
    st.caption(
        "ROC-AUC is optimistic here: the huge legit class keeps the false positive rate tiny. "
        f"The test set has only {res['split_sizes']['test'][1]} frauds, so differences between "
        "close models are within noise (see the intervals below)."
    )

    pr, roc = st.columns(2)
    for col, key, x, y, title in (
        (pr, "pr_curve", "recall", "precision", "Precision-recall (test)"),
        (roc, "roc_curve", "fpr", "tpr", "ROC (test)"),
    ):
        fig = go.Figure()
        for m in order:
            curve = models[m][key]
            deployed = m == res["deployed"]
            fig.add_scatter(
                x=curve[x],
                y=curve[y],
                mode="lines",
                name=models[m]["label"],
                line={"width": 4 if deployed else 1.5},
            )
        fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=480, legend_font_size=9)
        col.plotly_chart(fig, width="stretch")

    st.subheader("Bootstrap 95% confidence intervals")
    rows = [
        {
            "Model": models[m]["label"],
            "Test PR-AUC": (
                f"{models[m]['test_pr_auc']:.3f} [{c['pr_auc'][0]:.3f}, {c['pr_auc'][1]:.3f}]"
            ),
            "Recall @ 90% precision": (
                f"{models[m]['recall_at_precision']:.3f} "
                f"[{c['recall_at_precision'][0]:.3f}, {c['recall_at_precision'][1]:.3f}]"
            ),
        }
        for m, c in res["cis"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _apply_val_optimal(review_cost: float) -> None:
    """Button callback: set the slider to the threshold that minimises cost on validation."""
    _, splits, _ = load_data()
    _, s_val, _ = score_split("val")
    thr, _ = best_threshold(
        splits["val"][TARGET].to_numpy(), s_val, splits["val"]["Amount"].to_numpy(), review_cost
    )
    st.session_state["thr"] = float(thr)


def page_threshold() -> None:
    _, splits, _ = load_data()
    art = load_bundle()
    _, s_test, _ = score_split("test")
    y, amount = splits["test"][TARGET].to_numpy(), splits["test"]["Amount"].to_numpy()
    st.session_state.setdefault("thr", float(art.threshold))

    a, b = st.columns(2)
    review_cost = a.slider(
        "Review cost per flagged transaction (an assumption)",
        0.5,
        50.0,
        CFG["cost"]["review_cost"],
        0.5,
    )
    thr = b.slider("Decision threshold", 0.0, 1.0, step=0.001, format="%.3f", key="thr")
    b.button(
        "Use validation-optimal threshold",
        on_click=_apply_val_optimal,
        args=(review_cost,),
    )

    flagged = s_test >= thr
    op = operating_point(y, flagged)
    cost = total_cost(y, flagged, amount, review_cost)
    base = flag_nothing_cost(y, amount)
    saved, pct = savings(cost, base)
    half_cost = total_cost(y, s_test >= DEFAULT_THRESHOLD, amount, review_cost)

    m = st.columns(5)
    m[0].metric("Precision", f"{op['precision']:.3f}")
    m[1].metric("Recall", f"{op['recall']:.3f}")
    m[2].metric("Flagged", f"{op['n_flagged']:,}")
    m[3].metric("Total cost", f"{cost:,.0f}")
    m[4].metric("Cost saved", f"{saved:,.0f}", f"{pct:.1f}% vs flag-nothing")
    st.caption(
        f"Flag-nothing costs {base:,.0f}; a plain 0.5 threshold costs {half_cost:,.0f}. "
        f"Confusion at this threshold: TP {op['tp']}, FP {op['fp']}, "
        f"FN {op['fn']}, TN {op['tn']:,}."
    )

    curve = cost_by_threshold(y, s_test, amount, review_cost)
    curve = curve[curve["threshold"] > 0]
    fig = px.line(curve, x="threshold", y="cost", log_x=True, log_y=True)
    fig.add_vline(
        x=max(thr, curve["threshold"].min()), line_dash="dash", annotation_text="threshold"
    )
    fig.add_hline(y=base, line_dash="dot", annotation_text="flag nothing")
    fig.update_layout(title="Test cost vs threshold", height=420)
    st.plotly_chart(fig, width="stretch")
    st.info(
        "This page evaluates the test set for any slider value, to explore the trade-off. "
        "The production threshold is chosen on validation (the button above), not by moving "
        "the slider until the test cost looks good. Review cost is an assumption, not measured."
    )


def page_explorer() -> None:
    _, splits, _ = load_data()
    art = load_bundle()
    X, s, raw = score_split("test")
    test = splits["test"]
    # equal calibrated scores are common (isotonic plateaus), so break ties by the raw score
    order = np.lexsort((-raw, -s))
    top_n = st.slider("Rows to show (highest score first)", 25, 1000, 200, 25)
    pick = order[:top_n]
    table = pd.DataFrame(
        {
            "rank": np.arange(1, len(pick) + 1),
            "score": s[pick],
            "flagged": s[pick] >= art.threshold,
            "actual fraud": test[TARGET].to_numpy()[pick].astype(bool),
            "Amount": test["Amount"].to_numpy()[pick],
            "Time (h)": (test["Time"].to_numpy()[pick] / SECONDS_PER_HOUR).round(2),
        }
    )
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="explorer_table",
        column_config={"score": st.column_config.NumberColumn(format="%.4f")},
    )
    selected = event.selection.rows
    if not selected:
        st.info("Select a row to see why the model scored it that way.")
        return
    row = int(pick[selected[0]])
    contrib = pd.DataFrame(top_k_contributions(art.explainer, X.iloc[[row]], k=10))
    contrib["direction"] = np.where(contrib["contribution"] > 0, "raises risk", "lowers risk")
    fig = px.bar(
        contrib.iloc[::-1],
        x="contribution",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"raises risk": "#c44e52", "lowers risk": "#4c72b0"},
        hover_data=["value"],
    )
    fig.update_layout(title=f"Top 10 SHAP contributions, rank {selected[0] + 1}", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Log-odds contributions to the raw XGBoost score (before calibration). V1-V28 are "
        "anonymised PCA components, so they are named, not interpreted. amount_log and hour are "
        "standardised values."
    )


def _replay_window(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rows to replay: from the start, or centred on the k-th fraud (so frauds appear)."""
    mode = st.radio("Window", ["From the start", "Around a fraud"], horizontal=True)
    start = 0
    if mode == "Around a fraud":
        fraud_pos = np.flatnonzero(frame[TARGET].to_numpy() == 1)
        k = st.number_input("Fraud number", 1, len(fraud_pos), 1)
        start = max(int(fraud_pos[int(k) - 1]) - n // 2, 0)
    return frame.iloc[start : start + n]


def _draw_replay(results: list[dict[str, Any]], slots: dict[str, Any]) -> None:
    df = pd.DataFrame(results)
    flagged = int(df["is_fraud"].sum())
    caught = int((df["is_fraud"] & (df["Class"] == 1)).sum())
    frauds = int((df["Class"] == 1).sum())
    with slots["metrics"].container():
        m = st.columns(6)
        m[0].metric("Processed", f"{len(df):,}")
        m[1].metric("Flagged", f"{flagged:,}")
        m[2].metric("Alert rate", f"{flagged / len(df):.2%}")
        m[3].metric("Frauds caught", f"{caught} of {frauds}")
        m[4].metric("Cost saved (cumulative)", f"{df['cost_saved'].sum():,.0f}")
        m[5].metric("p95 round trip", f"{df['round_trip_ms'].quantile(0.95):.0f} ms")
    slots["chart"].line_chart(df["cost_saved"].cumsum().rename("cumulative cost saved"))
    feed = df.tail(FEED_ROWS).iloc[::-1][
        ["Time", "Amount", "fraud_probability", "is_fraud", "Class", "cost_saved"]
    ]
    slots["feed"].dataframe(
        feed.rename(columns={"is_fraud": "flagged", "Class": "actual fraud"}),
        hide_index=True,
        width="stretch",
    )


def page_replay() -> None:
    _, splits, _ = load_data()
    st.write("Streams test transactions, in time order, through the running API.")
    url = st.text_input("API URL", os.environ.get("API_URL", "http://127.0.0.1:8000"))
    c1, c2 = st.columns(2)
    n = c1.slider("Transactions", 50, 2000, 300, 50)
    speed = c2.select_slider(
        "Speed (x real time, 0 = as fast as possible)", [0, 100, 500, 2000, 10000], value=0
    )
    rows = _replay_window(splits["test"].reset_index(drop=True), n)
    if not st.button("Start replay"):
        return
    try:
        httpx.get(f"{url}/health", timeout=3.0).raise_for_status()
    except httpx.HTTPError:
        st.error(f"Cannot reach the API at {url}. Start it with `make serve`.")
        return
    slots = {"metrics": st.empty(), "chart": st.empty(), "feed": st.empty()}
    results: list[dict[str, Any]] = []
    try:
        with httpx.Client(base_url=url, timeout=10.0) as client:
            for result in replay(client, rows, speed=speed):
                results.append(result)
                if len(results) % UPDATE_EVERY == 0:
                    _draw_replay(results, slots)
    except httpx.HTTPError as exc:
        st.error(f"Replay stopped: {exc}")
    if results:
        _draw_replay(results, slots)
        summary = latency_summary(results)
        st.success(
            f"Done: {len(results)} transactions. API p95 {summary['server_p95_ms']:.1f} ms, "
            f"client round trip p95 {summary['round_trip_p95_ms']:.1f} ms."
        )
        st.caption(
            "The round trip includes this dashboard and your browser competing for the same "
            "machine, so it overstates API latency. The controlled benchmark "
            f"(target p95 < {summary['target_p95_ms']} ms) is in reports/results.md."
        )


# ---------------------------------------------------------------- main
st.title("FraudLens")
page = st.sidebar.radio("Page", PAGES)
{
    "Overview": page_overview,
    "Model comparison": page_models,
    "Threshold and cost": page_threshold,
    "Transaction explorer": page_explorer,
    "Live replay": page_replay,
}[page]()
