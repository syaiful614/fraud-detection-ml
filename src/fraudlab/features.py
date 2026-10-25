"""Feature engineering that ships *inside* the model pipeline (no training/serving skew).

The raw dataset exposes `Time` (seconds since the first transaction), 28 anonymised PCA
components (`V1..V28`) and `Amount`. Only `Time` and `Amount` can be engineered:

* `log_amount`: log1p tames the heavy right tail of transaction amounts.
* `hour_sin` / `hour_cos`: cyclical time-of-day. Raw `Time` is deliberately NOT used, because
  it is a non-stationary counter that a temporal hold-out can never see again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraudlab.data import V_COLUMNS

SECONDS_PER_DAY = 86_400
ENGINEERED = ["log_amount", "hour_sin", "hour_cos"]
FEATURE_NAMES = [*V_COLUMNS, *ENGINEERED]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless transformer: raw transaction frame -> model features (DataFrame)."""

    def fit(self, X: pd.DataFrame, y=None) -> FeatureBuilder:
        self._check(X)
        self.feature_names_out_ = list(FEATURE_NAMES)
        return self

    @staticmethod
    def _check(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureBuilder expects a pandas DataFrame with raw columns")
        missing = [c for c in ["Time", "Amount", *V_COLUMNS] if c not in X.columns]
        if missing:
            raise ValueError(f"Missing raw columns: {missing}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check(X)
        out = X[V_COLUMNS].astype("float64").copy()
        out["log_amount"] = np.log1p(X["Amount"].astype("float64").clip(lower=0))
        seconds_of_day = np.mod(X["Time"].to_numpy(dtype="float64"), SECONDS_PER_DAY)
        angle = 2.0 * np.pi * seconds_of_day / SECONDS_PER_DAY
        out["hour_sin"] = np.sin(angle)
        out["hour_cos"] = np.cos(angle)
        return out[FEATURE_NAMES]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURE_NAMES)
