import json

import pytest
from fastapi.testclient import TestClient

from fraudlab.artifacts import load_bundle
from fraudlab.config import Config
from fraudlab.data import RAW_FEATURES
from fraudlab.pipeline import run
from fraudlab.serve.app import create_app


@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_df):
    root = tmp_path_factory.mktemp("run")
    cfg = Config.from_dict(
        {
            "models_dir": str(root / "models"),
            "reports_dir": str(root / "reports"),
            "candidates": ["dummy", "logreg", "lgbm"],
            "split": {"cv_splits": 3, "cv_gap_rows": 50},
            "tuning": {"n_trials": 2},
            "evaluation": {"bootstrap_samples": 30, "alert_budgets": [0.01], "cv_alert_rate": 0.01},
        }
    )
    metrics = run(cfg, df=synthetic_df)
    return root, metrics


def test_pipeline_writes_all_artifacts(trained):
    root, metrics = trained
    for f in ("metrics.json", "results.md", "cv_folds.csv", "drift_psi.csv", "best_params.json"):
        assert (root / "reports" / f).exists(), f
    for fig in ("pr_curve_test", "shap_summary", "cost_curve", "cv_model_comparison"):
        assert (root / "reports" / "figures" / f"{fig}.png").exists(), fig
    assert (root / "models" / "fraud_model.joblib").exists()
    saved = json.loads((root / "reports" / "metrics.json").read_text())
    assert 0.0 <= saved["test"]["average_precision"] <= 1.0
    assert metrics["data"]["test_frauds"] > 0


def test_model_beats_trivial_baseline_on_learnable_data(trained):
    _, metrics = trained
    assert metrics["test"]["average_precision"] > 3 * metrics["test"]["prevalence"]


def test_net_savings_are_reported_against_no_model(trained):
    _, metrics = trained
    op = metrics["test"]["operating_points"]["cost_optimal"]
    assert op["baseline_cost"] == pytest.approx(op["total_cost"] + op["net_savings"])


@pytest.fixture(scope="module")
def client(trained):
    root, _ = trained
    with TestClient(create_app(root / "models")) as c:
        yield c


@pytest.fixture()
def payload(synthetic_df):
    return {k: float(v) for k, v in synthetic_df.iloc[10][RAW_FEATURES].items()}


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert 0 < info["threshold"] < 1


def test_predict_returns_probability_and_decision(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud_alert"] == (body["fraud_probability"] >= body["threshold"])
    assert body["top_factors"] is None


def test_predict_with_explanation(client, payload):
    body = client.post("/predict?explain=true", json=payload).json()
    assert len(body["top_factors"]) == 5
    assert {"feature", "contribution"} <= set(body["top_factors"][0])


def test_batch_prediction(client, payload):
    r = client.post("/predict/batch", json={"transactions": [payload] * 3})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_api_rejects_invalid_input(client, payload):
    assert (
        client.post("/predict", json={k: v for k, v in payload.items() if k != "V1"}).status_code
        == 422
    )
    assert client.post("/predict", json={**payload, "Amount": -5}).status_code == 422
    assert client.post("/predict", json={**payload, "surprise": 1}).status_code == 422
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_service_reports_503_without_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as c:
        assert c.get("/health").status_code == 503


def test_saved_bundle_roundtrip_matches_live_scores(trained, payload):
    import pandas as pd

    root, _ = trained
    bundle = load_bundle(root / "models")
    frame = pd.DataFrame([payload])[RAW_FEATURES]
    p1 = bundle.model.predict_proba(frame)[0, 1]
    p2 = load_bundle(root / "models").model.predict_proba(frame)[0, 1]
    assert p1 == pytest.approx(p2)
