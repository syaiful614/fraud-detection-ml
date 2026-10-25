"""Time-aware cross-validation that also returns out-of-fold (OOF) scores."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from fraudlab.metrics import average_precision, precision_recall_at_alert_rate, roc_auc
from fraudlab.models import build_model

log = logging.getLogger(__name__)


def cross_validate(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    n_jobs: int,
    alert_rate: float,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit a fresh pipeline per fold (so scalers, SMOTE, etc. never see validation data).

    Returns per-fold metrics and an array of OOF scores (NaN for rows never validated, i.e. the
    oldest block that only ever serves as training data).
    """
    oof = np.full(len(y), np.nan)
    rows = []
    for fold, (tr, va) in enumerate(splits):
        t0 = time.perf_counter()
        model = build_model(name, seed=seed, n_jobs=n_jobs, params=params, y_train=y.iloc[tr])
        model.fit(X.iloc[tr], y.iloc[tr])
        score = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = score
        pr = precision_recall_at_alert_rate(y.iloc[va], score, alert_rate)
        rows.append(
            {
                "model": name,
                "fold": fold,
                "n_train": len(tr),
                "n_val": len(va),
                "val_frauds": int(y.iloc[va].sum()),
                "average_precision": average_precision(y.iloc[va], score),
                "roc_auc": roc_auc(y.iloc[va], score),
                "recall_at_alert_rate": pr["recall"],
                "precision_at_alert_rate": pr["precision"],
                "fit_seconds": time.perf_counter() - t0,
            }
        )
    return pd.DataFrame(rows), oof


def summarize(folds: pd.DataFrame) -> dict[str, Any]:
    g = folds.groupby("model")
    out = {}
    for name, df in g:
        out[name] = {
            "model": name,
            "ap_mean": float(df["average_precision"].mean()),
            "ap_std": float(df["average_precision"].std(ddof=0)),
            "roc_auc_mean": float(df["roc_auc"].mean()),
            "recall_at_alert_rate_mean": float(df["recall_at_alert_rate"].mean()),
            "fit_seconds_mean": float(df["fit_seconds"].mean()),
        }
    return out
