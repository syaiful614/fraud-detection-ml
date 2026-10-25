"""Orchestration: load -> split -> compare -> tune -> threshold -> evaluate -> explain -> persist.

Design rules that keep the numbers honest:
  * the test period is the last slice of the timeline and is touched exactly once, at the end;
  * model selection, tuning and threshold choice use only the development period;
  * every transformer lives inside the model pipeline, so it is re-fit per CV fold.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraudlab import plots
from fraudlab.artifacts import environment_info, save_bundle
from fraudlab.config import Config
from fraudlab.data import RAW_FEATURES, TARGET, load_raw, sha256_file
from fraudlab.drift import feature_drift
from fraudlab.explain import tree_shap
from fraudlab.features import FEATURE_NAMES
from fraudlab.metrics import (
    average_precision,
    best_threshold_by_cost,
    bootstrap_ci,
    confusion_at_threshold,
    cost_curve,
    evaluate_cost,
    precision_recall_at_alert_rate,
    roc_auc,
)
from fraudlab.models import build_model
from fraudlab.reporting import write_results_md
from fraudlab.splits import check_folds_have_positives, expanding_window_splits, temporal_holdout
from fraudlab.tuning import tune_lgbm
from fraudlab.validation import cross_validate, summarize

log = logging.getLogger(__name__)
BASELINES = ("logreg", "isolation_forest", "random_forest", "lgbm")


def _builtin(obj: Any) -> Any:
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)}")


def run(cfg: Config, df: pd.DataFrame | None = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    reports_dir = Path(cfg.reports_dir)
    fig_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    seed, n_jobs = cfg.seed, cfg.n_jobs

    # ------------------------------------------------------------------ 1. data & splits
    data_sha = None
    if df is None:
        raw_path = Path(cfg.data.raw_path)
        df = load_raw(raw_path, cfg.data.drop_duplicates)
        data_sha = sha256_file(raw_path)
    dev, test = temporal_holdout(df, cfg.split.test_fraction)
    X_dev, y_dev = dev[RAW_FEATURES], dev[TARGET]
    X_test, y_test = test[RAW_FEATURES], test[TARGET]
    splits = expanding_window_splits(len(dev), cfg.split.cv_splits, cfg.split.cv_gap_rows)
    check_folds_have_positives(y_dev, splits)
    log.info(
        "Data: %d rows, %d frauds | dev %d (%d frauds) | test %d (%d frauds)",
        len(df),
        df[TARGET].sum(),
        len(dev),
        y_dev.sum(),
        len(test),
        y_test.sum(),
    )
    plots.plot_eda(df, fig_dir / "eda_overview.png")
    plots.plot_validation_scheme(
        len(dev), len(test), splits, y_dev, fig_dir / "validation_scheme.png"
    )

    # ------------------------------------------------------------------ 2. model comparison
    fold_tables = []
    for name in cfg.candidates:
        log.info("Cross-validating %s", name)
        folds, _ = cross_validate(
            name,
            X_dev,
            y_dev,
            splits,
            seed=seed,
            n_jobs=n_jobs,
            alert_rate=cfg.evaluation.cv_alert_rate,
        )
        fold_tables.append(folds)
        log.info("  mean AP = %.4f", folds["average_precision"].mean())
    all_folds = pd.concat(fold_tables, ignore_index=True)
    all_folds.to_csv(reports_dir / "cv_folds.csv", index=False)
    cv_summary = sorted(summarize(all_folds).values(), key=lambda r: -r["ap_mean"])
    plots.plot_cv_comparison(cv_summary, float(y_dev.mean()), fig_dir / "cv_model_comparison.png")

    # ------------------------------------------------------------------ 3. tuning
    log.info("Tuning LightGBM with Optuna (%d trials)", cfg.tuning.n_trials)
    best_params, study = tune_lgbm(
        X_dev,
        y_dev,
        splits,
        n_trials=cfg.tuning.n_trials,
        seed=seed,
        n_jobs=n_jobs,
        alert_rate=cfg.evaluation.cv_alert_rate,
    )
    study.trials_dataframe().to_csv(reports_dir / "optuna_trials.csv", index=False)
    (reports_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    # ------------------------------------------------------------------ 4. threshold (dev only)
    tuned_folds, oof = cross_validate(
        "lgbm",
        X_dev,
        y_dev,
        splits,
        seed=seed,
        n_jobs=n_jobs,
        alert_rate=cfg.evaluation.cv_alert_rate,
        params=best_params,
    )
    seen = ~np.isnan(oof)
    y_oof, s_oof = y_dev.to_numpy()[seen], oof[seen]
    amount_oof = X_dev["Amount"].to_numpy()[seen]
    review_cost = cfg.cost.review_cost
    best = best_threshold_by_cost(y_oof, s_oof, amount_oof, review_cost)
    prec, rec, thr = precision_recall_curve(y_oof, s_oof)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    thresholds = {
        "cost_optimal": best["threshold"],
        "f1_optimal": float(thr[int(np.argmax(f1[:-1]))]),
        "default_0.5": 0.5,
    }
    chosen = thresholds["cost_optimal"]
    plots.plot_cost_curve(
        cost_curve(y_oof, s_oof, amount_oof, review_cost),
        len(y_oof),
        best,
        fig_dir / "cost_curve.png",
    )
    log.info("Cost-optimal threshold on OOF scores: %.4f (%d alerts)", chosen, best["alerts"])

    # ------------------------------------------------------------------ 5. final fit + test
    log.info("Fitting final model on the full development period")
    final = build_model("lgbm", seed=seed, n_jobs=n_jobs, params=best_params).fit(X_dev, y_dev)
    s_test = final.predict_proba(X_test)[:, 1]
    y_t, a_t = y_test.to_numpy(), X_test["Amount"].to_numpy()

    test_metrics = {
        "average_precision": average_precision(y_t, s_test),
        "roc_auc": roc_auc(y_t, s_test),
        "prevalence": float(y_t.mean()),
        "ci95": bootstrap_ci(
            y_t,
            s_test,
            a_t,
            chosen,
            review_cost,
            n_boot=cfg.evaluation.bootstrap_samples,
            seed=seed,
        ),
    }
    operating_points = {
        name: {
            **confusion_at_threshold(y_t, s_test, t),
            **evaluate_cost(y_t, s_test, a_t, t, review_cost),
        }
        for name, t in thresholds.items()
    }
    alert_budgets = [
        precision_recall_at_alert_rate(y_t, s_test, r) for r in cfg.evaluation.alert_budgets
    ]

    curves = {}
    p, r, _ = precision_recall_curve(y_t, s_test)
    curves["final (tuned LightGBM)"] = (p, r, test_metrics["average_precision"])
    baselines = {}
    for name in BASELINES:
        log.info("Baseline on test: %s", name)
        m = build_model(name, seed=seed, n_jobs=n_jobs, y_train=y_dev).fit(X_dev, y_dev)
        s = m.predict_proba(X_test)[:, 1]
        label = "lightgbm (defaults)" if name == "lgbm" else name
        baselines[label] = {
            "average_precision": average_precision(y_t, s),
            "roc_auc": roc_auc(y_t, s),
        }
        p, r, _ = precision_recall_curve(y_t, s)
        curves[label] = (p, r, baselines[label]["average_precision"])
    op = operating_points["cost_optimal"]
    plots.plot_pr_curves(
        curves,
        test_metrics["prevalence"],
        (op["recall"], op["precision"]),
        fig_dir / "pr_curve_test.png",
    )
    plots.plot_score_distribution(y_t, s_test, chosen, fig_dir / "score_distribution.png")

    # ------------------------------------------------------------------ 6. explain + drift
    rng = np.random.default_rng(seed)
    fraud_idx = np.flatnonzero(y_t == 1)
    legit_idx = np.flatnonzero(y_t == 0)
    sample = np.sort(
        np.concatenate([fraud_idx, rng.choice(legit_idx, min(3000, len(legit_idx)), replace=False)])
    )
    shap_values, _, shap_feats = tree_shap(final, X_test.iloc[sample])
    plots.plot_shap_summary(shap_values, shap_feats, fig_dir / "shap_summary.png")
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=shap_feats.columns)
    importance = importance.sort_values(ascending=False)

    fb = final.named_steps["features"]
    drift = feature_drift(fb.transform(X_dev), fb.transform(X_test))
    drift.to_csv(reports_dir / "drift_psi.csv", index=False)
    plots.plot_drift(drift, fig_dir / "drift_psi.png")

    # ------------------------------------------------------------------ 7. persist
    metrics: dict[str, Any] = {
        "data": {
            "rows": len(df),
            "frauds": int(df[TARGET].sum()),
            "fraud_rate": float(df[TARGET].mean()),
            "dev_rows": len(dev),
            "dev_frauds": int(y_dev.sum()),
            "test_rows": len(test),
            "test_frauds": int(y_test.sum()),
            "sha256": data_sha,
        },
        "cv_summary": cv_summary,
        "tuning": {
            "n_trials": cfg.tuning.n_trials,
            "best_cv_ap": float(study.best_value),
            "tuned_cv_ap_mean": float(tuned_folds["average_precision"].mean()),
            "best_params": best_params,
        },
        "threshold": {
            **best,
            "oof_confusion": confusion_at_threshold(y_oof, s_oof, chosen),
            "policy": "cost_optimal",
            "candidates": thresholds,
            "review_cost": review_cost,
        },
        "test": {
            **test_metrics,
            "baselines": baselines,
            "operating_points": operating_points,
            "alert_budgets": alert_budgets,
        },
        "explain": {"top_features": importance.head(10).round(4).to_dict()},
        "drift": drift.head(8).to_dict(orient="records"),
        "environment": environment_info(),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_builtin))
    write_results_md(metrics, reports_dir / "results.md")

    save_bundle(
        final,
        {
            "model": "lightgbm (tuned)",
            "threshold": chosen,
            "threshold_policy": "cost-optimal on out-of-fold scores (development period)",
            "review_cost": review_cost,
            "raw_features": RAW_FEATURES,
            "feature_names": FEATURE_NAMES,
            "best_params": best_params,
            "data_sha256": data_sha,
            "training_window_seconds": [float(dev["Time"].min()), float(dev["Time"].max())],
            "test_metrics": {k: test_metrics[k] for k in ("average_precision", "roc_auc")},
            "config": cfg.to_dict(),
        },
        cfg.models_dir,
    )
    log.info(
        "Done in %.1f s. Artefacts: %s, %s", metrics["runtime_seconds"], reports_dir, cfg.models_dir
    )
    return metrics
