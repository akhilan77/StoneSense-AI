"""FastAPI integration tests validating endpoint routing responses and schema boundaries."""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Verifies response structure of the public health endpoint."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["resnet18_ct_classification"] is True
    assert data["models_loaded"]["xgboost_risk_prediction"] is True


def test_models_info_endpoint(client: TestClient, auth_headers_developer):
    """Verifies response structure of models informational endpoints with authentication."""
    res = client.get("/api/v1/models", headers=auth_headers_developer)
    assert res.status_code == 200
    data = res.json()
    assert "dl_resnet18" in data
    assert "ml_xgboost" in data
    assert data["dl_resnet18"]["loaded"] is True


def test_predict_risk_endpoint(client: TestClient, auth_headers_hospital_1):
    """Verifies that predict risk route parses PatientInformation and returns classifications."""
    patient_payload = {
        "hospital_id": 1,
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
    res = client.post("/api/v1/predict/risk", json=patient_payload, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    data = res.json()
    assert "probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "High"]
    assert data["shap"]["top_features"]
    assert set(data["shap"]["feature_directions"].values()) <= {"increases", "decreases", "neutral"}


def test_predict_image_endpoint(client: TestClient, auth_headers_hospital_1):
    """Verifies upload processing of binary CT scans."""
    from pathlib import Path
    from PIL import Image
    import io
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    files = {"image": ("dummy_ct.png", img_bytes, "image/png")}
    data = {"hospital_id": 1}
    res = client.post("/api/v1/predict/image", files=files, data=data, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    resp_data = res.json()
    assert "class_name" in resp_data
    assert "confidence" in resp_data
    assert resp_data["class_name"] in ["Cyst", "Normal", "Stone", "Tumor"]
    if resp_data.get("gradcam") and resp_data["gradcam"].get("overlay_url"):
        assert resp_data["gradcam"]["overlay_url"].startswith("/static/gradcam/")
