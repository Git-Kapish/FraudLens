import json

import pytest
from conftest import make_artifacts, make_frame
from fastapi.testclient import TestClient

from fraud.api.main import create_app
from fraud.api.schemas import MAX_BATCH
from fraud.config import resolve_path
from fraud.data import FEATURES

THRESHOLD = 0.3


@pytest.fixture(scope="module")
def frame():
    return make_frame(n=3000, fraud_rate=0.1)


@pytest.fixture(scope="module")
def client(frame):
    """App wired to a tiny synthetic model, so tests need neither the dataset nor models/."""
    with TestClient(create_app(make_artifacts(frame, THRESHOLD))) as c:
        yield c


@pytest.fixture
def payload(frame):
    return frame.iloc[0][FEATURES].to_dict()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test-model", "threshold": THRESHOLD}


def test_predict_happy_path(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] == (body["fraud_probability"] >= THRESHOLD)
    assert body["threshold"] == THRESHOLD
    assert body["model_version"] == "test-model"
    assert body["latency_ms"] > 0


def test_missing_field_is_422(client, payload):
    del payload["V7"]
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_amount_is_422(client, payload):
    payload["Amount"] = -1.0
    assert client.post("/predict", json=payload).status_code == 422


def test_bad_types_and_unknown_fields_are_422_not_500(client, payload):
    assert client.post("/predict", json={**payload, "V1": "abc"}).status_code == 422
    assert client.post("/predict", json={**payload, "extra": 1}).status_code == 422
    assert (
        client.post(
            "/predict", content=b"{not json", headers={"content-type": "application/json"}
        ).status_code
        == 422
    )
    # httpx refuses to encode NaN, so send the raw (non-standard) JSON literal ourselves
    nan_body = json.dumps({**payload, "V2": float("nan")}).encode()
    nan = client.post("/predict", content=nan_body, headers={"content-type": "application/json"})
    assert nan.status_code == 422
    assert "input" not in nan.json()["detail"][0]  # payload values are never echoed back


def test_batch_happy_path_and_order(client, frame):
    rows = [frame.iloc[i][FEATURES].to_dict() for i in range(3)]
    r = client.post("/predict/batch", json={"transactions": rows})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    single = client.post("/predict", json=rows[1]).json()["fraud_probability"]
    assert body["results"][1]["fraud_probability"] == pytest.approx(single)


def test_batch_limit_and_empty_are_422(client, payload):
    too_many = {"transactions": [payload] * (MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
    assert (
        client.post("/predict/batch", json={"transactions": [payload] * MAX_BATCH}).status_code
        == 200
    )


def test_explain_returns_top5_signed_contributions(client, payload):
    r = client.post("/explain", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributions"]) == 5
    mags = [abs(c["contribution"]) for c in body["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert {"feature", "value", "contribution"} == set(body["contributions"][0])


def test_metrics_count_requests_and_report_percentiles(client, payload):
    before = client.get("/metrics").json()["latency"].get("/predict", {}).get("requests", 0)
    for _ in range(5):
        client.post("/predict", json=payload)
    snap = client.get("/metrics").json()
    assert snap["latency"]["/predict"]["requests"] == before + 5
    assert 0 < snap["latency"]["/predict"]["p50_ms"] <= snap["latency"]["/predict"]["p95_ms"]
    assert 0.0 <= snap["flag_rate"] <= 1.0


@pytest.mark.skipif(
    not (resolve_path("models_dir") / "threshold.json").exists(), reason="no trained artifacts"
)
def test_real_artifacts_load_and_score():
    row = {name: 0.0 for name in FEATURES}
    with TestClient(create_app()) as c:
        r = c.post("/predict", json=row)
        assert r.status_code == 200
        assert c.get("/health").json()["model_version"] == r.json()["model_version"]
