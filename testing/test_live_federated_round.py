"""Integration and Unit Tests for Live Federated Learning Round Control & Visualization."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

from app.main import app
from app.db.database import init_db, SessionLocal
from app.db.models import FederatedRound, HospitalTrainingRun, ModelVersion, Hospital
from app.services.federated_coordinator import federated_coordinator

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    init_db()


def test_start_federated_round_api():
    """Verify starting a round via API returns immediate 200 with round details."""
    response = client.post("/api/v1/developer/federated/rounds/start", json={"num_rounds": 1})
    assert response.status_code in [200, 409]
    if response.status_code == 200:
        data = response.json()
        assert "round" in data
        assert data["status"] == "started"
        assert "global_model_version" in data


def test_duplicate_round_prevention():
    """Verify that starting a duplicate round while one is running raises 409 Conflict."""
    # Force coordinator is_running state
    was_running = federated_coordinator.is_running
    federated_coordinator.is_running = True
    try:
        response = client.post("/api/v1/developer/federated/rounds/start", json={"num_rounds": 1})
        assert response.status_code == 409
        assert "currently running" in response.json()["detail"]
    finally:
        federated_coordinator.is_running = was_running


def test_current_round_status_api():
    """Verify GET /api/v1/developer/federated/rounds/current/status returns actual coordinator state."""
    response = client.get("/api/v1/developer/federated/rounds/current/status")
    assert response.status_code == 200
    data = response.json()
    assert "round" in data
    assert "status" in data
    assert "global_model_version" in data
    assert "clients" in data
    assert len(data["clients"]) >= 3


def test_hospital_live_status_endpoints():
    """Verify GET /{hospital_id}/federated-live-status for both integer and code identifiers."""
    # Numeric ID test
    res1 = client.get("/api/v1/hospital/1/federated-live-status")
    assert res1.status_code == 200
    d1 = res1.json()
    assert "hospital_id" in d1
    assert "status" in d1
    assert "global_model_version" in d1

    # Code identifier test on /hospitals route
    res2 = client.get("/api/v1/hospitals/HOSP-002/federated-live-status")
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["hospital_id"] == "HOSP-002"
    assert "local_training" in d2


def test_end_to_end_fl_execution_and_persistence():
    """Verify full local client training, FedAvg aggregation, model versioning, and DB persistence."""
    # If not running, initiate round
    if not federated_coordinator.is_running:
        federated_coordinator.start_round(
            num_rounds=1,
            local_epochs=1,
            batch_size=32,
            lr=0.0005,
            mode="iid",
            device="cpu"
        )

    # Wait for worker thread to complete execution
    max_wait = 90
    start_t = time.time()
    while federated_coordinator.is_running and (time.time() - start_t < max_wait):
        time.sleep(1)

    assert federated_coordinator.is_running is False
    assert federated_coordinator.status == "COMPLETED"
    assert federated_coordinator.error_message is None

    db = SessionLocal()
    try:
        target_round = federated_coordinator.current_round

        # Verify Database Records
        fed_round = db.query(FederatedRound).filter_by(round_number=target_round).first()
        assert fed_round is not None
        assert fed_round.status == "completed"
        assert fed_round.global_val_acc is not None
        assert fed_round.global_val_f1 is not None

        runs = db.query(HospitalTrainingRun).filter_by(round_id=fed_round.id).all()
        assert len(runs) == 3
        for run in runs:
            assert run.sample_count > 0
            assert run.train_acc is not None

        # Verify ModelVersion Checkpoint
        tag = f"resnet18_fed_round_{target_round:03d}"
        mv = db.query(ModelVersion).filter_by(version_tag=tag).first()
        assert mv is not None
        assert mv.is_deployed is True

        # Verify Checkpoint File on disk
        expected_ckpt = PROJECT_ROOT / "dl" / "models" / "federated" / f"{tag}.pth"
        assert expected_ckpt.exists()

    finally:
        db.close()

