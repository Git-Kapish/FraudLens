import pytest
from conftest import make_artifacts, make_frame
from fastapi.testclient import TestClient

from fraud.api.main import create_app
from fraud.replay import cost_saved, latency_summary, replay


def test_cost_saved_hand_computed():
    assert cost_saved(1, 100.0, True, 5.0) == 95.0  # caught fraud: loss avoided minus review
    assert cost_saved(0, 100.0, True, 5.0) == -5.0  # false alarm: pay the review
    assert cost_saved(1, 100.0, False, 5.0) == 0.0  # missed fraud: same as flag-nothing
    assert cost_saved(0, 100.0, False, 5.0) == 0.0


def test_latency_summary_percentiles():
    rows = [{"server_ms": float(i), "round_trip_ms": float(i + 1)} for i in range(1, 101)]
    s = latency_summary(rows)
    assert s["n_calls"] == 100
    assert s["server_p50_ms"] == pytest.approx(50.5)
    assert s["round_trip_p95_ms"] == pytest.approx(96.05)
    assert s["meets_target"] is False  # p95 is far above the 50 ms target


def test_replay_streams_in_order_through_the_api():
    frame = make_frame(n=300, fraud_rate=0.1).sort_values("Time")
    with TestClient(create_app(make_artifacts(frame))) as client:
        out = list(replay(client, frame.iloc[:10], review_cost=5.0))
    assert [r["position"] for r in out] == list(range(10))
    assert [r["Time"] for r in out] == sorted(r["Time"] for r in out)
    assert all(0.0 <= r["fraud_probability"] <= 1.0 and r["round_trip_ms"] > 0 for r in out)
    for r in out:  # per-row cost saving follows the rule, not just the sign
        assert r["cost_saved"] == cost_saved(r["Class"], r["Amount"], r["is_fraud"], 5.0)
