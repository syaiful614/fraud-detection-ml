"""Explainability via TreeSHAP values computed natively by LightGBM (`pred_contrib=True`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def tree_shap(model: Pipeline, X_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (shap_values, base_value, engineered_features) in log-odds space.

    For every row: base_value + sum(shap_values) = raw model output (log-odds of fraud).
    """
    features = model.named_steps["features"].transform(X_raw)
    booster = model.named_steps["clf"].booster_
    contrib = booster.predict(features, pred_contrib=True)
    return contrib[:, :-1], contrib[:, -1], features


def top_factors(contrib_row: np.ndarray, names: list[str], k: int = 5) -> list[dict]:
    """The k features that pushed this prediction the hardest (either direction)."""
    order = np.argsort(-np.abs(contrib_row))[:k]
    return [{"feature": names[i], "contribution": float(contrib_row[i])} for i in order]
