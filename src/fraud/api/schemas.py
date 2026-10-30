"""Pydantic v2 request and response models for the scoring API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud.config import load_config
from fraud.data import FEATURES

MAX_BATCH = load_config()["api"]["max_batch_size"]


def _transaction_fields() -> dict[str, tuple[type[float], object]]:
    """All 30 raw features, required and finite; Amount must be >= 0."""
    fields = {}
    for name in FEATURES:
        extra = {"ge": 0} if name == "Amount" else {}
        fields[name] = (float, Field(..., allow_inf_nan=False, **extra))
    return fields


Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    __doc__="One transaction: Time, V1-V28 (anonymised PCA components) and Amount.",
    **_transaction_fields(),
)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool


class PredictResponse(Prediction):
    threshold: float
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: list[Prediction]
    threshold: float
    model_version: str
    latency_ms: float


class Contribution(BaseModel):
    feature: str
    value: float
    contribution: float


class ExplainResponse(PredictResponse):
    base_value: float
    contributions: list[Contribution]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    threshold: float
