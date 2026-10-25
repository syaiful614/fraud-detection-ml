"""Bayesian hyper-parameter search (Optuna TPE) with time-aware CV as the objective."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd

from fraudlab.validation import cross_validate

log = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),
    }


def tune_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    *,
    n_trials: int,
    seed: int,
    n_jobs: int,
    alert_rate: float,
) -> tuple[dict[str, Any], optuna.Study]:
    """Maximise mean average precision across expanding-window folds."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        folds, _ = cross_validate(
            "lgbm", X, y, splits, seed=seed, n_jobs=n_jobs, alert_rate=alert_rate, params=params
        )
        trial.set_user_attr("fold_ap", [float(v) for v in folds["average_precision"]])
        return float(np.mean(folds["average_precision"]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def log_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        log.info(
            "  trial %02d  CV AP=%.4f  (best so far %.4f)",
            trial.number + 1,
            trial.value,
            study.best_value,
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[log_progress])
    return dict(study.best_params), study
