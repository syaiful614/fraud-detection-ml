import numpy as np
import pytest

from fraudlab.drift import psi
from fraudlab.metrics import (
    best_threshold_by_cost,
    bootstrap_ci,
    confusion_at_threshold,
    cost_curve,
    evaluate_cost,
    precision_recall_at_alert_rate,
)

# Hand-computed example: total fraud amount = 150
Y = np.array([1, 0, 1, 0])
S = np.array([0.9, 0.8, 0.3, 0.1])
A = np.array([100.0, 10.0, 50.0, 5.0])


def test_cost_curve_matches_hand_calculation():
    curve = cost_curve(Y, S, A, review_cost=20)
    # k = 0..4 alerts -> 150, 70, 90, 60, 80
    np.testing.assert_allclose(curve.cost, [150, 70, 90, 60, 80])


def test_best_threshold_by_cost():
    best = best_threshold_by_cost(Y, S, A, review_cost=20)
    assert best["alerts"] == 3
    assert best["cost"] == pytest.approx(60)
    assert best["threshold"] == pytest.approx(0.2)  # midway between 0.3 and 0.1
    assert best["baseline_cost"] == pytest.approx(150)


def test_expensive_reviews_mean_flag_nothing():
    best = best_threshold_by_cost(Y, S, A, review_cost=1000)
    assert best["alerts"] == 0
    assert best["threshold"] > S.max()


def test_ties_are_never_split():
    s = np.array([0.5, 0.5, 0.1])
    curve = cost_curve(np.array([1, 0, 0]), s, np.array([10.0, 1.0, 1.0]), 1.0)
    assert np.isnan(curve.thresholds[1])  # cannot flag exactly one of two tied scores


def test_evaluate_cost_and_confusion():
    out = evaluate_cost(Y, S, A, threshold=0.2, review_cost=20)
    assert out["total_cost"] == pytest.approx(60)
    assert out["net_savings"] == pytest.approx(90)
    conf = confusion_at_threshold(Y, S, 0.2)
    assert (conf["tp"], conf["fp"], conf["fn"], conf["tn"]) == (2, 1, 0, 1)
    assert conf["recall"] == 1.0


def test_alert_budget_metrics():
    out = precision_recall_at_alert_rate(Y, S, rate=0.5)
    assert out["alerts"] == 2
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    s = np.clip(0.5 * y + rng.normal(0.2, 0.15, 3000), 0, 1)
    a = rng.lognormal(3, 1, 3000)
    ci = bootstrap_ci(y, s, a, 0.5, 5.0, n_boot=150, seed=1)
    for lo, hi in ci.values():
        assert lo <= hi
    from fraudlab.metrics import average_precision

    lo, hi = ci["average_precision"]
    assert lo <= average_precision(y, s) <= hi


def test_psi_detects_shift_and_ignores_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=20_000)) < 0.05
    assert psi(ref, rng.normal(loc=2.0, size=20_000)) > 0.25
