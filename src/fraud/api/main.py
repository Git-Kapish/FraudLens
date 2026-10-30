"""Scoring service. Model, threshold and explainer are loaded once at startup (lifespan)."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fraud.api.schemas import (
    BatchRequest,
    BatchResponse,
    ExplainResponse,
    HealthResponse,
    Prediction,
    PredictResponse,
    Transaction,
)
from fraud.config import resolve_path
from fraud.data import FEATURES
from fraud.explain import base_estimator, build_explainer, top_k_contributions

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fraud.api")
LATENCY_WINDOW = 10_000  # most recent requests kept per endpoint for percentiles


@dataclass
class Artifacts:
    """Everything the service needs to score a request."""

    pipeline: Any
    model: Any
    explainer: Any
    threshold: float
    model_version: str


def load_artifacts() -> Artifacts:
    """Load the frozen model bundle from models/ (files written by train/evaluate)."""
    models_dir = resolve_path("models_dir")
    meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    threshold = json.loads((models_dir / "threshold.json").read_text(encoding="utf-8"))
    # joblib is pickle: these files are written by this repo's own train.py, so they are trusted
    pipeline = joblib.load(models_dir / "pipeline.joblib")
    model = joblib.load(models_dir / "best_calibrated.joblib")
    base_estimator(model).set_params(n_jobs=1)  # thread fan-out only slows single-row calls
    return Artifacts(
        pipeline, model, build_explainer(model), threshold["threshold"], meta["model_version"]
    )


class Metrics:
    """In-memory request counters and latency percentiles, safe across worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, int] = defaultdict(int)
        self._latency: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=LATENCY_WINDOW))
        self.scored = 0
        self.flagged = 0

    def record(self, endpoint: str, n_scored: int, n_flagged: int, latency_ms: float) -> None:
        with self._lock:
            self._requests[endpoint] += 1
            self._latency[endpoint].append(latency_ms)
            self.scored += n_scored
            self.flagged += n_flagged

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latency = {
                ep: {
                    "requests": self._requests[ep],
                    "p50_ms": float(np.percentile(v, 50)),
                    "p95_ms": float(np.percentile(v, 95)),
                }
                for ep, v in self._latency.items()
            }
            return {
                "requests_total": sum(self._requests.values()),
                "transactions_scored": self.scored,
                "transactions_flagged": self.flagged,
                "flag_rate": self.flagged / self.scored if self.scored else 0.0,
                "latency": latency,
            }


def _frame(transactions: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([t.model_dump() for t in transactions], columns=FEATURES)


def _score(art: Artifacts, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = art.pipeline.transform(df)
    return X, art.model.predict_proba(X)[:, 1]


def _log(endpoint: str, **fields: Any) -> None:
    """One JSON line per request: ids, scores and latency only, never feature values."""
    logger.info(json.dumps({"request_id": uuid.uuid4().hex[:8], "endpoint": endpoint, **fields}))


def create_app(artifacts: Artifacts | None = None) -> FastAPI:
    """Build the app. Tests pass small in-memory `artifacts`; production loads from models/."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.art = artifacts or load_artifacts()
        app.state.metrics = Metrics()
        yield

    app = FastAPI(title="FraudLens scoring API", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_payload(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The default body echoes `input`; a NaN there cannot be rendered as JSON and became a
        # 500. Report where and why only, which also never echoes payload values.
        errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        art = request.app.state.art
        return HealthResponse(status="ok", model_version=art.model_version, threshold=art.threshold)

    @app.post("/predict", response_model=PredictResponse)
    def predict(tx: Transaction, request: Request) -> PredictResponse:
        start = time.perf_counter()
        art = request.app.state.art
        prob = float(_score(art, _frame([tx]))[1][0])
        flagged = prob >= art.threshold
        ms = (time.perf_counter() - start) * 1000
        request.app.state.metrics.record("/predict", 1, int(flagged), ms)
        _log("/predict", score=round(prob, 6), flagged=flagged, latency_ms=round(ms, 2))
        return PredictResponse(
            fraud_probability=prob,
            is_fraud=flagged,
            threshold=art.threshold,
            model_version=art.model_version,
            latency_ms=ms,
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
        start = time.perf_counter()
        art = request.app.state.art
        probs = _score(art, _frame(body.transactions))[1]
        flagged = probs >= art.threshold
        ms = (time.perf_counter() - start) * 1000
        request.app.state.metrics.record("/predict/batch", len(probs), int(flagged.sum()), ms)
        _log(
            "/predict/batch",
            n=len(probs),
            flagged=int(flagged.sum()),
            max_score=round(float(probs.max()), 6),
            latency_ms=round(ms, 2),
        )
        return BatchResponse(
            results=[
                Prediction(fraud_probability=float(p), is_fraud=bool(f))
                for p, f in zip(probs, flagged, strict=True)
            ],
            threshold=art.threshold,
            model_version=art.model_version,
            latency_ms=ms,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(tx: Transaction, request: Request) -> ExplainResponse:
        start = time.perf_counter()
        art = request.app.state.art
        X, probs = _score(art, _frame([tx]))
        prob = float(probs[0])
        flagged = prob >= art.threshold
        top = top_k_contributions(art.explainer, X)
        ms = (time.perf_counter() - start) * 1000
        request.app.state.metrics.record("/explain", 1, int(flagged), ms)
        _log("/explain", score=round(prob, 6), flagged=flagged, latency_ms=round(ms, 2))
        return ExplainResponse(
            fraud_probability=prob,
            is_fraud=flagged,
            threshold=art.threshold,
            model_version=art.model_version,
            latency_ms=ms,
            base_value=float(art.explainer.expected_value),
            contributions=top,
        )

    @app.get("/metrics")
    def metrics(request: Request) -> dict[str, Any]:
        return request.app.state.metrics.snapshot()

    return app


app = create_app()
