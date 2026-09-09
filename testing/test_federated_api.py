import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.main import app
from app.db.database import init_db

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()


def test_federated_overview_endpoint():
    response = client.get("/api/v1/developer/federated-overview")
    assert response.status_code == 200
    data = response.json()
    assert "current_round" in data
    assert "global_accuracy" in data
    assert "global_f1" in data
    assert "active_hospitals_count" in data


def test_round_history_endpoint():
    response = client.get("/api/v1/developer/round-history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_hospital_participation_endpoint():
    response = client.get("/api/v1/developer/hospital-participation")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "hospital_code" in data[0]
        assert "sample_contribution_pct" in data[0]


def test_hospital_dataset_status_endpoint():
    # Test for hospital 1
    response = client.get("/api/v1/hospital/1/dataset-status")
    assert response.status_code == 200
    data = response.json()
    assert "class_distribution" in data
    assert "dataset_size" in data


def test_hospital_current_model_endpoint():
    response = client.get("/api/v1/hospital/1/current-model")
    assert response.status_code == 200
    data = response.json()
    assert "current_model_version" in data
    assert "global_accuracy" in data
