"""FastAPI integration tests validating endpoint routing responses and schema boundaries."""

import json
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Verifies response structure of the health endpoint."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["resnet18_ct_classification"] is True
    assert data["models_loaded"]["xgboost_risk_prediction"] is True


def test_models_info_endpoint(client: TestClient):
    """Verifies response structure of models informational endpoints."""
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "dl_resnet18" in data
    assert "ml_xgboost" in data
    assert data["dl_resnet18"]["loaded"] is True


def test_predict_risk_endpoint(client: TestClient):
    """Verifies that predict risk route parses PatientInformation and returns classifications."""
    # PatientInformation request schema payload
    patient_payload = {
        "age": 45,
        "gender": "male",
        "bmi": 24.5,
        "blood_pressure": 120.0,
        "diabetes": False,
        "family_history": True,
        "water_intake": 2.5,
        "urine_ph": 6.0,
        "urine_specific_gravity": 1.015,
        "calcium": 4.5,
        "uric_acid": 3.0,
        "creatinine": 1.0
    }
    res = client.post("/api/v1/predict/risk", json=patient_payload)
    assert res.status_code == 200
    data = res.json()
    assert "probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "High"]


def test_predict_image_endpoint(client: TestClient):
    """Verifies upload processing of binary CT scans."""
    from PIL import Image
    import io
    img = Image.new("L", (224, 224), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    files = {"image": ("dummy_ct.png", img_bytes, "image/png")}
    res = client.post("/api/v1/predict/image", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "class_name" in data
    assert "confidence" in data
    assert data["class_name"] in ["Cyst", "Normal", "Stone", "Tumor"]


def test_assessment_endpoint(client: TestClient):
    """Verifies orchestrator builds aggregated multi-modal assessments."""
    patient_payload = {
        "age": 45,
        "gender": "male",
        "bmi": 24.5,
        "blood_pressure": 120.0,
        "diabetes": False,
        "family_history": True,
        "water_intake": 2.5,
        "urine_ph": 6.0,
        "urine_specific_gravity": 1.015,
        "calcium": 4.5,
        "uric_acid": 3.0,
        "creatinine": 1.0
    }

    from PIL import Image
    import io
    img = Image.new("L", (224, 224), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    form_data = {
        "patient_data": json.dumps(patient_payload)
    }
    files = {"image": ("dummy_ct.png", img_bytes, "image/png")}

    res = client.post("/api/v1/assessment", data=form_data, files=files)
    assert res.status_code == 200
    data = res.json()
    
    assert "risk_prediction" in data
    assert "ct_prediction" in data
    assert "shap" in data
    assert "recommendation" in data
