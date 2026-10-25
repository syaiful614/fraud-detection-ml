"""Model zoo. Every model is a full sklearn/imblearn Pipeline that starts with FeatureBuilder."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudlab.features import FeatureBuilder


class IsolationForestScorer(ClassifierMixin, BaseEstimator):
    """Unsupervised anomaly detector exposed through the classifier API.

    Labels are ignored during fit. Scores are min-max scaled into [0, 1] so that ranking metrics
    (average precision, ROC-AUC) can be compared with supervised models.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X)
        raw = -self.model_.score_samples(X)
        self.lo_, self.hi_ = np.percentile(raw, [0.1, 99.9])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        raw = -self.model_.score_samples(X)
        score = np.clip((raw - self.lo_) / max(self.hi_ - self.lo_, 1e-12), 0.0, 1.0)
        return np.column_stack([1.0 - score, score])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def lgbm_defaults(seed: int, n_jobs: int) -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "max_bin": 63,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbose": -1,
    }


def build_lgbm(seed: int, n_jobs: int, params: dict[str, Any] | None = None) -> LGBMClassifier:
    return LGBMClassifier(**{**lgbm_defaults(seed, n_jobs), **(params or {})})


def build_model(
    name: str,
    *,
    seed: int = 42,
    n_jobs: int = -1,
    params: dict[str, Any] | None = None,
    y_train: pd.Series | None = None,
) -> Pipeline:
    """Create an unfitted pipeline.

    `y_train` is only needed by `lgbm_weighted` to derive the class-imbalance ratio from the
    *training fold* (never from data it will be evaluated on).
    """
    fb = ("features", FeatureBuilder())
    if name == "dummy":
        return Pipeline([fb, ("clf", DummyClassifier(strategy="prior"))])
    if name == "logreg":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
        return Pipeline([fb, ("scale", StandardScaler()), ("clf", clf)])
    if name == "isolation_forest":
        clf = IsolationForestScorer(random_state=seed, n_jobs=n_jobs)
        return Pipeline([fb, ("scale", StandardScaler()), ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            max_samples=0.5,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )
        return Pipeline([fb, ("clf", clf)])
    if name == "lgbm":
        return Pipeline([fb, ("clf", build_lgbm(seed, n_jobs, params))])
    if name == "lgbm_weighted":
        if y_train is None:
            raise ValueError("lgbm_weighted needs y_train to compute the class ratio")
        pos = max(int(y_train.sum()), 1)
        ratio = float(np.sqrt((len(y_train) - pos) / pos))  # square-root heuristic
        merged = {"scale_pos_weight": ratio, **(params or {})}
        return Pipeline([fb, ("clf", build_lgbm(seed, n_jobs, merged))])
    if name == "lgbm_smote":
        smote = SMOTE(sampling_strategy=0.05, k_neighbors=5, random_state=seed)
        return ImbPipeline([fb, ("smote", smote), ("clf", build_lgbm(seed, n_jobs, params))])
    raise ValueError(f"Unknown model '{name}'")
