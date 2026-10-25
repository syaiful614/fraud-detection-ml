"""FastAPI scoring service. The saved pipeline contains feature engineering, so callers send
raw transaction fields and training/serving skew is impossible by construction."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraudlab import __version__
from fraudlab.artifacts import ModelBundle, load_bundle
from fraudlab.data import RAW_FEATURES, V_COLUMNS
from fraudlab.explain import top_factors, tree_shap

log = logging.getLogger("fraudlab.api")
MAX_BATCH = 1000

_example = {"Time": 406.0, **{c: 0.0 for c in V_COLUMNS}, "Amount": 149.62}
_fields = {
    "Time": (float, Field(..., ge=0, description="Seconds since the first transaction")),
    **{c: (float, Field(..., description="Anonymised PCA component")) for c in V_COLUMNS},
    "Amount": (float, Field(..., ge=0, description="Transaction amount")),
}
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": _example}),
    **_fields,
)


class Factor(BaseModel):
    feature: str
    contribution: float = Field(description="SHAP value in log-odds; positive pushes towards fraud")


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud_alert: bool
    threshold: float
    top_factors: list[Factor] | None = None


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


def _score(bundle: ModelBundle, transactions: list, explain: bool) -> list[Prediction]:
    frame = pd.DataFrame([t.model_dump() for t in transactions])[RAW_FEATURES]
    proba = bundle.model.predict_proba(frame)[:, 1]
    factors = [None] * len(frame)
    if explain:
        shap_values, _, feats = tree_shap(bundle.model, frame)
        names = list(feats.columns)
        factors = [top_factors(row, names) for row in shap_values]
    return [
        Prediction(
            fraud_probability=float(p),
            is_fraud_alert=bool(p >= bundle.threshold),
            threshold=bundle.threshold,
            top_factors=f,
        )
        for p, f in zip(proba, factors, strict=True)
    ]


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    directory = Path(model_dir or os.environ.get("MODEL_DIR", "models"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.bundle = load_bundle(directory)
            log.info("Model loaded from %s", directory)
        except FileNotFoundError as exc:
            app.state.bundle = None
            log.error("%s", exc)
        yield

    app = FastAPI(
        title="Fraud Detection API",
        version=__version__,
        description="Scores card transactions. Alerts are raised at a cost-optimal threshold.",
        lifespan=lifespan,
    )

    def bundle() -> ModelBundle:
        b = app.state.bundle
        if b is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run `fraudlab train`.")
        return b

    @app.get("/health")
    def health() -> dict[str, str]:
        bundle()
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict:
        m = bundle().metadata
        keys = (
            "model",
            "threshold",
            "threshold_policy",
            "review_cost",
            "trained_at",
            "git_commit",
            "test_metrics",
            "environment",
            "training_window_seconds",
        )
        return {k: m.get(k) for k in keys}

    @app.post("/predict", response_model=Prediction)
    def predict(tx: Transaction, explain: bool = Query(False, description="Add top SHAP factors")):  # type: ignore[valid-type]
        return _score(bundle(), [tx], explain)[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest, explain: bool = Query(False)):
        return BatchResponse(predictions=_score(bundle(), req.transactions, explain))

    return app


app = create_app()
