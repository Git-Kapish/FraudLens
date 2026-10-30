"""Cost model: false negative costs the Amount, every flagged transaction costs a review.

review_cost is an assumption (config.yaml), not a measured number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_cost(y: np.ndarray, flagged: np.ndarray, amount: np.ndarray, review_cost: float) -> float:
    """FN = Amount lost, FP = TP = review_cost, TN = 0."""
    y, flagged, amount = np.asarray(y), np.asarray(flagged, dtype=bool), np.asarray(amount)
    missed = amount[(y == 1) & ~flagged].sum()
    return float(missed + review_cost * flagged.sum())


def flag_nothing_cost(y: np.ndarray, amount: np.ndarray) -> float:
    """Cost of never flagging: every fraud's Amount is lost."""
    return float(np.asarray(amount)[np.asarray(y) == 1].sum())


def cost_by_threshold(
    y: np.ndarray, scores: np.ndarray, amount: np.ndarray, review_cost: float
) -> pd.DataFrame:
    """Total cost of the rule `flag if score >= t` at every distinct score, plus flag-nothing.

    Vectorised: sort once, then use suffix sums of fraud Amount.
    """
    y, scores, amount = np.asarray(y), np.asarray(scores, dtype=float), np.asarray(amount)
    n = len(scores)
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    fraud_amount = (amount * (y == 1))[order]
    suffix = np.append(np.cumsum(fraud_amount[::-1])[::-1], 0.0)  # suffix[i] = sum(fa[i:])
    thr, first = np.unique(s, return_index=True)
    n_flagged = n - first
    caught = suffix[first]
    # a threshold just above the top score flags nothing
    thr = np.append(thr, np.nextafter(s[-1], np.inf))
    n_flagged = np.append(n_flagged, 0)
    caught = np.append(caught, 0.0)
    total_fraud = fraud_amount.sum()
    cost = review_cost * n_flagged + (total_fraud - caught)
    return pd.DataFrame({"threshold": thr, "n_flagged": n_flagged, "cost": cost})


def best_threshold(
    y: np.ndarray, scores: np.ndarray, amount: np.ndarray, review_cost: float
) -> tuple[float, float]:
    """Threshold minimising total cost, and that cost. Ties go to the higher threshold."""
    table = cost_by_threshold(y, scores, amount, review_cost)
    idx = np.flatnonzero(table["cost"].to_numpy() == table["cost"].min())[-1]
    return float(table["threshold"].iloc[idx]), float(table["cost"].iloc[idx])


def savings(cost: float, baseline: float) -> tuple[float, float]:
    """Absolute and percent saving versus a baseline cost (0% if the baseline is 0)."""
    saved = baseline - cost
    return float(saved), float(100.0 * saved / baseline) if baseline else 0.0
