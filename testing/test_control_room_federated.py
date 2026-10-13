"""Comprehensive test suite for Federated Learning Control Room truthful state,
events persistence, tenant isolation, and recovery.
"""

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
from app.db.models import FederatedRound, HospitalTrainingRun, ModelVersion, Hospital, RoundEvent
from app.services.federated_coordinator import federated_coordinator
from app.core.security import create_access_token

admin_token = create_access_token(data={"sub": "admin@stonesense.ai", "role": "admin", "hospital_id": None})
dev_token = create_access_token(data={"sub": "dev@stonesense.ai", "role": "developer", "hospital_id": None})
hosp1_token = create_access_token(data={"sub": "hospital1@stonesense.ai", "role": "hospital_user", "hospital_id": 1})
hosp2_token = create_access_token(data={"sub": "hospital2@stonesense.ai", "role": "hospital_user", "hospital_id": 2})

admin_client = TestClient(app, headers={"Authorization": f"Bearer {admin_token}"})
dev_client = TestClient(app, headers={"Authorization": f"Bearer {dev_token}"})
hosp1_client = TestClient(app, headers={"Authorization": f"Bearer {hosp1_token}"})
hosp2_client = TestClient(app, headers={"Authorization": f"Bearer {hosp2_token}"})


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment(tmp_path_factory):
    from dl.federated.model_manager import model_manager
    test_models_dir = tmp_path_factory.mktemp("fed_control_models")
    model_manager.set_models_dir(test_models_dir)
    init_db()


def test_current_round_events_route():
    """Verify GET /api/v1/developer/federated/rounds/current/events returns ordered events."""
    db = SessionLocal()
    try:
        # Seed test event
        target_round = federated_coordinator.current_round
        if not target_round:
            latest = db.query(FederatedRound).order_by(FederatedRound.round_number.desc()).first()
            target_round = latest.round_number if latest else 1
        evt = RoundEvent(
            round_number=target_round,
            event_type="CONTROL_ROOM_TEST_EVENT",
            stage="TEST_STAGE",
            hospital_code="HOSP-001",
            data={"test_key": "truthful_telemetry"},
        )
        db.add(evt)
        db.commit()

        res = dev_client.get("/api/v1/developer/federated/rounds/current/events")
        assert res.status_code == 200
        events = res.json()
        assert isinstance(events, list)
        matching = [e for e in events if e["event_type"] == "CONTROL_ROOM_TEST_EVENT"]
        assert len(matching) > 0
        assert matching[0]["data"]["test_key"] == "truthful_telemetry"
    finally:
        db.close()


def test_round_status_recovery_on_idle_or_restart():
    """Verify that when coordinator is idle, get_live_round_status hydrates real metrics from DB."""
    db = SessionLocal()
    try:
        # Create a historical round in DB
        test_r_num = 999
        existing = db.query(FederatedRound).filter_by(round_number=test_r_num).first()
        if not existing:
            fed_round = FederatedRound(
                round_number=test_r_num,
                mode="iid",
                selected_hospital_ids=[1, 2, 3],
                participants_count=3,
                global_train_loss=0.075,
                global_train_acc=0.965,
                global_val_loss=0.082,
                global_val_acc=0.971,
                global_val_f1=0.968,
                global_val_precision=0.970,
                global_val_recall=0.969,
                duration_sec=14.5,
                status="completed",
            )
            db.add(fed_round)
            db.flush()

            # Add runs
            for hid, hcode in [(1, "HOSP-001"), (2, "HOSP-002"), (3, "HOSP-003")]:
                db.add(HospitalTrainingRun(
                    round_id=fed_round.id,
                    round_number=test_r_num,
                    hospital_id=hid,
                    hospital_code=hcode,
                    train_loss=0.07,
                    train_acc=0.96,
                    train_f1=0.95,
                    val_loss=0.08,
                    val_acc=0.97,
                    val_f1=0.96,
                    sample_count=1000,
                    duration_sec=3.2,
                ))

            # Add candidate model version
            db.add(ModelVersion(
                model_family="resnet18_ct",
                version_tag=f"resnet18_fed_round_{test_r_num:03d}",
                round_id=fed_round.id,
                accuracy=0.971,
                f1_score=0.968,
                is_deployed=False,
                status="pending_review",
                artifact_path=f"dl/models/federated/resnet18_fed_round_{test_r_num:03d}.pth",
            ))
            db.commit()

        # Query specific historical round via API
        res = dev_client.get(f"/api/v1/developer/federated/rounds/{test_r_num}/status")
        assert res.status_code == 200
        data = res.json()
        assert data["round"] == test_r_num
        assert data["status"] == "COMPLETED"
        assert data["global_model_version"] == f"resnet18_fed_round_{test_r_num:03d}"
        assert data["metrics"]["accuracy"] == 0.971
        assert data["metrics"]["f1"] == 0.968
        assert data["is_deployed"] is False
        assert len(data["clients"]) >= 3
        # Ensure sample counts were faithfully reconstructed
        for c in data["clients"]:
            assert c["samples"] == 1000
            assert c["accuracy"] is not None

    finally:
        db.close()


def test_failed_round_persistence_in_db():
    """Verify that when a round execution throws an error, the round is persisted with status='failed'."""
    # Attempt to trigger a round with invalid min_participants via coordinator directly or worker exception
    test_fail_round = 888
    db = SessionLocal()
    try:
        # Clean existing test round if present
        db.query(FederatedRound).filter_by(round_number=test_fail_round).delete()
        db.commit()

        # Simulate exception inside worker workflow
        try:
            federated_coordinator._run_round_workflow(
                round_num=test_fail_round,
                local_epochs=1,
                batch_size=32,
                lr=0.0005,
                mode="iid",
                device_str="cpu",
                selected_hospital_codes=["INVALID_CODE"],
                selected_hospital_ids=[9999],
                min_participants=2,
            )
        except Exception:
            pass

        # Check DB to confirm status="failed" is persisted
        failed_record = db.query(FederatedRound).filter_by(round_number=test_fail_round).first()
        assert failed_record is not None
        assert failed_record.status == "failed"

    finally:
        db.close()


def test_hospital_tenant_isolation_security():
    """Verify cross-tenant security: Hospital 1 cannot read Hospital 2's live telemetry or data."""
    # Hospital 1 queries Hospital 1 -> 200 OK
    res_own = hosp1_client.get("/api/v1/hospital/1/federated-live-status")
    assert res_own.status_code == 200
    assert res_own.json()["hospital_id"] == "HOSP-001"

    # Hospital 1 attempts to query Hospital 2 -> 403 Forbidden
    res_other = hosp1_client.get("/api/v1/hospital/2/federated-live-status")
    assert res_other.status_code == 403
    assert "Access forbidden" in res_other.json()["detail"] or "cannot access" in res_other.json()["detail"]

    # Developer cannot query patient private history
    res_dev_history = dev_client.get("/api/v1/hospital/1/history")
    assert res_dev_history.status_code == 403
    assert "Developers are not permitted" in res_dev_history.json()["detail"]


def test_no_automatic_model_deployment_guarantee():
    """Verify that all federated round model versions are created with is_deployed=False."""
    db = SessionLocal()
    try:
        fed_versions = db.query(ModelVersion).filter(ModelVersion.version_tag.like("resnet18_fed_round_%")).all()
        for mv in fed_versions:
            # Under the safety gate constraint, no federated round can auto-deploy
            assert mv.is_deployed is False, f"Model version {mv.version_tag} was unexpectedly marked is_deployed=True"
    finally:
        db.close()
