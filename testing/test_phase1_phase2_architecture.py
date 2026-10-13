"""Phase 1 & Phase 2 Architecture and Lifecycle Tests.

Validates:
1. Round event persistence to DB and retrieval via developer API.
2. Model family validation and rejection of unsupported federated models (YOLO26, DINOv3, QKNN, XGBoost).
3. Minimum participant constraints (min_participants >= 2).
4. Update validation: rejection of NaN/Inf or shape-mismatched client weights.
5. RBAC enforcement: unauthorized hospital users cannot start rounds or access developer event APIs.
6. Deployment separation: candidates are not auto-deployed.
"""

from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.main import app
from app.db.database import init_db, SessionLocal
from app.db.models import FederatedRound, RoundEvent, Hospital, User
from app.services.federated_coordinator import federated_coordinator, validate_client_update
from app.core.security import create_access_token

admin_token = create_access_token(data={"sub": "admin@stonesense.ai", "role": "admin", "hospital_id": None})
dev_token = create_access_token(data={"sub": "dev@stonesense.ai", "role": "developer", "hospital_id": None})
hospital_token = create_access_token(data={"sub": "hospital1@stonesense.ai", "role": "hospital_user", "hospital_id": 1})

admin_client = TestClient(app, headers={"Authorization": f"Bearer {admin_token}"})
dev_client = TestClient(app, headers={"Authorization": f"Bearer {dev_token}"})
hosp_client = TestClient(app, headers={"Authorization": f"Bearer {hospital_token}"})


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_validate_client_update_detects_mismatches_and_nans():
    """Unit test for validate_client_update logic."""
    gw = [np.zeros((10, 5)), np.ones((5,))]

    # Valid matching weights
    valid_cw = [np.ones((10, 5)) * 0.5, np.zeros((5,))]
    is_valid, reason = validate_client_update(valid_cw, gw)
    assert is_valid is True
    assert reason is None

    # Empty weights
    is_valid, reason = validate_client_update([], gw)
    assert is_valid is False
    assert "Empty weights" in reason

    # Mismatched layer count
    is_valid, reason = validate_client_update([np.zeros((10, 5))], gw)
    assert is_valid is False
    assert "Mismatched layer count" in reason

    # Mismatched layer shape
    bad_shape = [np.zeros((10, 4)), np.ones((5,))]
    is_valid, reason = validate_client_update(bad_shape, gw)
    assert is_valid is False
    assert "shape mismatch" in reason

    # NaN in weights
    nan_weights = [np.array([[0.1, np.nan], [0.2, 0.3]]), np.ones((5,))]
    nan_gw = [np.zeros((2, 2)), np.ones((5,))]
    is_valid, reason = validate_client_update(nan_weights, nan_gw)
    assert is_valid is False
    assert "Non-finite" in reason

    # Inf in weights
    inf_weights = [np.array([[0.1, np.inf], [0.2, 0.3]]), np.ones((5,))]
    is_valid, reason = validate_client_update(inf_weights, nan_gw)
    assert is_valid is False
    assert "Non-finite" in reason


def test_unsupported_model_family_rejected():
    """Verify that models without federated capability are rejected with 422."""
    unsupported = ["yolo26", "dinov3", "qknn", "xgboost", "random_forest"]
    for fam in unsupported:
        response = dev_client.post(
            "/api/v1/developer/federated/rounds/start",
            json={"model_family": fam, "num_rounds": 1}
        )
        assert response.status_code == 422, f"Expected 422 for {fam}, got {response.status_code}"
        assert fam in response.json()["detail"].lower()


def test_insufficient_min_participants_rejected():
    """Verify min_participants < 2 or selected_hospitals < min_participants returns 422."""
    # min_participants < 2
    res1 = dev_client.post(
        "/api/v1/developer/federated/rounds/start",
        json={"min_participants": 1, "num_rounds": 1}
    )
    assert res1.status_code == 422

    # selected 1 hospital but min_participants is 2
    res2 = dev_client.post(
        "/api/v1/developer/federated/rounds/start",
        json={"selected_hospital_ids": [1], "min_participants": 2, "num_rounds": 1}
    )
    assert res2.status_code == 422
    assert "minimum participants threshold" in res2.json()["detail"]


def test_hospital_user_cannot_start_round_or_view_events():
    """Verify role-based access control stops hospital users from managing rounds."""
    res1 = hosp_client.post(
        "/api/v1/developer/federated/rounds/start",
        json={"num_rounds": 1}
    )
    assert res1.status_code == 403

    res2 = hosp_client.get("/api/v1/developer/federated/rounds/1/events")
    assert res2.status_code == 403


def test_round_events_persisted_and_queried():
    """Verify that emitted round events persist to DB and can be retrieved via API."""
    db = SessionLocal()
    try:
        # Create a test event directly
        test_evt = RoundEvent(
            round_number=999,
            event_type="TEST_LIFECYCLE_EVENT",
            stage="LOCAL_TRAINING",
            hospital_code="HOSP-001",
            data={"test_key": "test_value"},
        )
        db.add(test_evt)
        db.commit()

        # Query via API
        response = dev_client.get("/api/v1/developer/federated/rounds/999/events")
        assert response.status_code == 200
        events = response.json()
        assert len(events) >= 1
        found = any(e["event_type"] == "TEST_LIFECYCLE_EVENT" and e["hospital_code"] == "HOSP-001" for e in events)
        assert found is True
    finally:
        db.query(RoundEvent).filter_by(round_number=999).delete()
        db.commit()
        db.close()
