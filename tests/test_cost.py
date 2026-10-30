import numpy as np
import pytest

from fraud.cost import best_threshold, cost_by_threshold, flag_nothing_cost, savings, total_cost

Y = np.array([1, 1, 0, 0])
AMOUNT = np.array([100.0, 50.0, 20.0, 30.0])


def test_total_cost_hand_computed():
    # TP (row 0) costs 5, FN (row 1) costs its Amount 50, FP (row 2) costs 5, TN costs 0
    flagged = np.array([1, 0, 1, 0])
    assert total_cost(Y, flagged, AMOUNT, review_cost=5.0) == 60.0


def test_flag_nothing_and_flag_everything():
    assert flag_nothing_cost(Y, AMOUNT) == 150.0
    assert total_cost(Y, np.zeros(4), AMOUNT, 5.0) == 150.0
    assert total_cost(Y, np.ones(4), AMOUNT, 5.0) == 20.0  # 2 TP + 2 FP, 4 reviews


def test_cost_by_threshold_matches_total_cost_for_every_row():
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    table = cost_by_threshold(Y, scores, AMOUNT, 5.0)
    for t, cost in zip(table["threshold"], table["cost"], strict=True):
        assert cost == pytest.approx(total_cost(Y, scores >= t, AMOUNT, 5.0))


def test_best_threshold_hand_computed():
    # flag top1: 5+50=55, top2: 10, top3: 15, all: 20, none: 150
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    assert best_threshold(Y, scores, AMOUNT, 5.0) == (0.8, 10.0)


def test_ties_are_flagged_together():
    scores = np.array([0.5, 0.5, 0.1, 0.1])
    t, cost = best_threshold(Y, scores, AMOUNT, 5.0)
    assert t == 0.5 and cost == 10.0


def test_high_review_cost_flags_nothing():
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    t, cost = best_threshold(Y, scores, AMOUNT, review_cost=1000.0)
    assert t > scores.max()
    assert cost == flag_nothing_cost(Y, AMOUNT)


def test_savings():
    assert savings(60.0, 150.0) == (90.0, 60.0)
    assert savings(10.0, 0.0) == (-10.0, 0.0)
