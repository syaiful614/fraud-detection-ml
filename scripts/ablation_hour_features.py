"""Ablation: does the time-of-day feature help, or does it only memorise the 2-day window?

Fits the tuned LightGBM on the development period with and without hour_sin/hour_cos and
compares average precision on the untouched test period and on the CV folds.
Usage: python scripts/ablation_hour_features.py
"""

import json
import warnings

import numpy as np
from sklearn.pipeline import Pipeline

from fraudlab.config import Config
from fraudlab.data import RAW_FEATURES, TARGET, load_raw
from fraudlab.features import FEATURE_NAMES, FeatureBuilder
from fraudlab.metrics import average_precision
from fraudlab.models import build_lgbm
from fraudlab.splits import expanding_window_splits, temporal_holdout

warnings.filterwarnings("ignore")


class NoHour(FeatureBuilder):
    def transform(self, X):
        return super().transform(X).drop(columns=["hour_sin", "hour_cos"])


def make(builder, params, seed):
    return Pipeline([("features", builder), ("clf", build_lgbm(seed, -1, params))])


cfg = Config.load("configs/default.yaml")
with open("reports/best_params.json") as fh:
    params = json.load(fh)
dev, test = temporal_holdout(load_raw(cfg.data.raw_path), cfg.split.test_fraction)
X, y = dev[RAW_FEATURES], dev[TARGET]
splits = expanding_window_splits(len(dev), cfg.split.cv_splits, cfg.split.cv_gap_rows)

variants = (
    ("with hour features", FeatureBuilder(), len(FEATURE_NAMES)),
    ("without hour features", NoHour(), len(FEATURE_NAMES) - 2),
)
for label, builder, n_features in variants:
    cv = []
    for tr, va in splits:
        m = make(builder, params, cfg.seed).fit(X.iloc[tr], y.iloc[tr])
        cv.append(average_precision(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    m = make(builder, params, cfg.seed).fit(X, y)
    ap = average_precision(test[TARGET], m.predict_proba(test[RAW_FEATURES])[:, 1])
    print(f"{label:24s} CV AP {np.mean(cv):.3f} | test AP {ap:.3f} ({n_features} features)")
