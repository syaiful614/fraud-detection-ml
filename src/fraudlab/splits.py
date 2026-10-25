"""Time-aware splitting. Random K-fold would let the model peek at the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def temporal_holdout(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last `test_fraction` of the timeline becomes the test set.

    `df` must already be sorted by `Time`.
    """
    if not df["Time"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by Time before a temporal split")
    cut = int(round(len(df) * (1.0 - test_fraction)))
    dev, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    if dev["Time"].max() > test["Time"].min():
        raise AssertionError("Temporal leakage: development data overlaps the test period")
    return dev, test


def expanding_window_splits(
    n_rows: int, n_splits: int, gap: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: train on the past, validate on the next block.

    `gap` rows between train and validation are purged to mimic label latency
    (real fraud labels arrive days later, so the most recent rows are not yet trustworthy).
    """
    return list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.arange(n_rows)))


def check_folds_have_positives(y: pd.Series, splits) -> None:
    for i, (_, val_idx) in enumerate(splits):
        if int(y.iloc[val_idx].sum()) == 0:
            raise ValueError(f"CV fold {i} has no fraud cases; reduce cv_splits")
