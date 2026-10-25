"""Population Stability Index (PSI): a standard production drift check."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def psi(expected, actual, bins: int = 10) -> float:
    """PSI of `actual` against `expected`, using quantile bins of `expected`.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = [(c, psi(reference[c], current[c], bins)) for c in reference.columns]
    df = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    df["status"] = pd.cut(
        df["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "major"]
    ).astype(str)
    return df.reset_index(drop=True)
