"""Replay test transactions through the scoring API in time order.

Used by scripts/replay.py (latency measurement) and the dashboard's live replay page.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import httpx
import numpy as np
import pandas as pd

from fraud.config import load_config
from fraud.data import FEATURES, TARGET


def cost_saved(is_fraud: int, amount: float, flagged: bool, review_cost: float) -> float:
    """Cost saved versus flag-nothing for one transaction.

    Flag-nothing loses `amount` on a fraud. Flagging costs `review_cost`, and a caught
    fraud avoids the loss. A missed fraud or an unflagged legit row saves nothing.
    """
    if not flagged:
        return 0.0
    return (amount if is_fraud else 0.0) - review_cost


def replay(
    client: httpx.Client, rows: pd.DataFrame, speed: float = 0.0, review_cost: float | None = None
) -> Iterator[dict[str, Any]]:
    """POST each row to /predict in order, yielding one result dict per transaction.

    `speed` compresses real time: 1000 means 1000 transaction-seconds per wall second.
    0 sends as fast as possible. `rows` must carry the 30 features plus Class.
    """
    review_cost = load_config()["cost"]["review_cost"] if review_cost is None else review_cost
    previous_time = None
    for position, (_, row) in enumerate(rows.iterrows()):
        if speed > 0 and previous_time is not None:
            time.sleep(max(row["Time"] - previous_time, 0.0) / speed)
        previous_time = row["Time"]
        start = time.perf_counter()
        response = client.post("/predict", json=row[FEATURES].to_dict())
        round_trip_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        body = response.json()
        yield {
            "position": position,
            "Time": float(row["Time"]),
            "Amount": float(row["Amount"]),
            "Class": int(row[TARGET]),
            "fraud_probability": body["fraud_probability"],
            "is_fraud": body["is_fraud"],
            "server_ms": body["latency_ms"],
            "round_trip_ms": round_trip_ms,
            "cost_saved": cost_saved(
                int(row[TARGET]), float(row["Amount"]), body["is_fraud"], review_cost
            ),
        }


def latency_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """p50/p95/max of server-side and client round-trip latency, plus the target."""
    server = np.array([r["server_ms"] for r in results])
    trip = np.array([r["round_trip_ms"] for r in results])
    target = load_config()["api"]["p95_latency_target_ms"]
    return {
        "n_calls": len(results),
        "server_p50_ms": float(np.percentile(server, 50)),
        "server_p95_ms": float(np.percentile(server, 95)),
        "round_trip_p50_ms": float(np.percentile(trip, 50)),
        "round_trip_p95_ms": float(np.percentile(trip, 95)),
        "round_trip_max_ms": float(trip.max()),
        "target_p95_ms": target,
        "meets_target": bool(np.percentile(trip, 95) < target),
    }
