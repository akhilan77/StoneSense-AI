"""API Regression Tests ensuring zero breaking changes to /api/v1/predict/risk contract."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "backend" / "app"))
sys.path.append(str(PROJECT_ROOT / "ml"))

from app.main import app
from app.services.model_loader import model_loader


@pytest.fixture(scope="module")
def api_client():
    model_loader.load_all_models()
    with TestClient(app) as c:
        yield c


def test_predict_risk_response_keys_regression(api_client):
    """Verifies that /api/v1/predict/risk exact response schema and keys are 100% preserved."""
    payload = {
        "age": 48,
        "gender": "male",
        "bmi": 26.2,
        "blood_pressure": 125.0,
        "diabetes": False,
        "family_history": True,
        "water_intake": 2.0,
        "urine_ph": 6.1,
        "urine_specific_gravity": 1.018,
        "calcium": 5.0,
        "uric_acid": 3.2,
        "creatinine": 1.1
    }

    res = api_client.post("/api/v1/predict/risk", json=payload)
    assert res.status_code == 200, f"Endpoint returned error: {res.text}"

    data = res.json()

    # Assert all top-level keys
    expected_top_keys = {"probability", "risk_level", "confidence", "inference_time_sec", "shap"}
    assert expected_top_keys.issubset(data.keys()), f"Missing keys in response: {expected_top_keys - set(data.keys())}"

    # Assert types and value bounds
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in ["Low", "High"]
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["inference_time_sec"], float)

    # Assert shap sub-schema
    shap_data = data["shap"]
    assert isinstance(shap_data, dict)
    expected_shap_keys = {"top_features", "feature_contributions", "feature_directions", "summary"}
    assert expected_shap_keys.issubset(shap_data.keys()), f"Missing shap keys: {expected_shap_keys - set(shap_data.keys())}"

    assert isinstance(shap_data["top_features"], list)
    assert len(shap_data["top_features"]) == 6
    assert isinstance(shap_data["feature_contributions"], dict)
    assert len(shap_data["feature_contributions"]) == 6
    assert isinstance(shap_data["feature_directions"], dict)
    assert set(shap_data["feature_directions"].values()).issubset({"increases", "decreases", "neutral"})
    assert isinstance(shap_data["summary"], str)


def test_health_check_includes_models(api_client):
    """Verifies health endpoint reports healthy status with loaded model singletons."""
    res = api_client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["resnet18_ct_classification"] is True
    assert data["models_loaded"]["xgboost_risk_prediction"] is True
    assert data["models_loaded"]["preprocessing_pipeline"] is True
