"""Metrics for rare-event detection: ranking quality, alert budgets and money.

Accuracy is meaningless at a 0.17% base rate (predicting "legit" always scores 99.83%), so the
primary metric is average precision (area under the precision-recall curve).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def average_precision(y, s) -> float:
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def confusion_at_threshold(y, s, threshold: float) -> dict[str, float]:
    y = np.asarray(y)
    flagged = np.asarray(s) >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alerts": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def precision_recall_at_alert_rate(y, s, rate: float) -> dict[str, float]:
    """Precision/recall when analysts can only review the top `rate` share of transactions."""
    y = np.asarray(y)
    k = max(1, math.ceil(rate * len(y)))
    top = np.argsort(-np.asarray(s), kind="stable")[:k]
    tp = int(y[top].sum())
    return {
        "alert_rate": float(rate),
        "alerts": int(k),
        "tp": tp,
        "precision": tp / k,
        "recall": tp / max(int(y.sum()), 1),
    }


@dataclass
class CostCurve:
    alerts: np.ndarray  # number of transactions flagged (k = 0..n)
    cost: np.ndarray  # total cost if the top-k scores are flagged
    thresholds: np.ndarray  # score threshold that flags exactly the top-k (nan if k is a tie)


def cost_curve(y, s, amount, review_cost: float) -> CostCurve:
    """Total cost for every possible alert count.

    cost(k) = (amount of frauds NOT in the top-k) + review_cost * k
    i.e. a missed fraud loses its amount; every alert costs analyst time; caught fraud costs 0.
    """
    y, s, amount = np.asarray(y), np.asarray(s, dtype=float), np.asarray(amount, dtype=float)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    fraud_amount = np.where(y == 1, amount, 0.0)[order]
    caught = np.concatenate([[0.0], np.cumsum(fraud_amount)])
    k = np.arange(len(y) + 1)
    cost = (fraud_amount.sum() - caught) + review_cost * k

    thresholds = np.full(len(y) + 1, np.nan)
    thresholds[0] = np.nextafter(s_sorted[0], np.inf)  # flag nothing
    thresholds[len(y)] = s_sorted[-1]  # flag everything
    if len(y) > 1:
        mid = (s_sorted[:-1] + s_sorted[1:]) / 2.0
        boundary = s_sorted[:-1] > s_sorted[1:]  # only cut between distinct scores
        thresholds[1:-1] = np.where(boundary, mid, np.nan)
    return CostCurve(alerts=k, cost=cost, thresholds=thresholds)


def best_threshold_by_cost(y, s, amount, review_cost: float) -> dict[str, float]:
    curve = cost_curve(y, s, amount, review_cost)
    masked = np.where(np.isnan(curve.thresholds), np.inf, curve.cost)
    best = int(np.argmin(masked))
    return {
        "threshold": float(curve.thresholds[best]),
        "alerts": int(curve.alerts[best]),
        "cost": float(curve.cost[best]),
        "baseline_cost": float(curve.cost[0]),
    }


def evaluate_cost(y, s, amount, threshold: float, review_cost: float) -> dict[str, float]:
    """Monetary outcome of using `threshold` versus having no model at all."""
    y, amount = np.asarray(y), np.asarray(amount, dtype=float)
    flagged = np.asarray(s) >= threshold
    missed = float(amount[(y == 1) & ~flagged].sum())
    caught = float(amount[(y == 1) & flagged].sum())
    reviews = float(review_cost * flagged.sum())
    total = missed + reviews
    baseline = float(amount[y == 1].sum())  # no model: every fraud goes through
    return {
        "total_cost": total,
        "missed_fraud_amount": missed,
        "caught_fraud_amount": caught,
        "review_cost_total": reviews,
        "baseline_cost": baseline,
        "net_savings": baseline - total,
        "net_savings_pct": (baseline - total) / baseline if baseline else 0.0,
    }


def bootstrap_ci(
    y,
    s,
    amount,
    threshold: float,
    review_cost: float,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs. The test set has few frauds, so uncertainty must be reported."""
    y, s, amount = np.asarray(y), np.asarray(s, dtype=float), np.asarray(amount, dtype=float)
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {
        "average_precision": [],
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "net_savings": [],
    }
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb, ab = y[idx], s[idx], amount[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        conf = confusion_at_threshold(yb, sb, threshold)
        out["average_precision"].append(average_precision(yb, sb))
        out["roc_auc"].append(roc_auc(yb, sb))
        out["precision"].append(conf["precision"])
        out["recall"].append(conf["recall"])
        out["net_savings"].append(evaluate_cost(yb, sb, ab, threshold, review_cost)["net_savings"])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in out.items()}
