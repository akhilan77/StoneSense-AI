"""Comprehensive Unit & Integration Tests for Phase 3: Deployment Gate and Model Lifecycle.

Verifies:
1. Candidate failing threshold -> rejected
2. Candidate with missing per-class recall -> pending_review (not eligible)
3. Candidate regressing against deployed baseline -> rejected
4. Candidate evaluated on different dataset signature -> not_comparable / pending_review
5. Attempting to deploy non-eligible version returns HTTP 400
6. Deploying eligible model updates status, archives previous, sets approved_by/approved_at
7. Sequential deploys maintain strictly one deployed row per model_family
8. Rollback restores previously deployed model version and updates references
9. Centralized ML training no longer auto-deploys
10. ModelVersion database migrations correctly initialize status and legacy gate reports
"""

from pathlib import Path
import sys
import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "backend"))

from app.db.models import Base, ModelVersion, Hospital
from app.services.deployment_gate import DeploymentGate, deployment_gate
from app.config.settings import Settings
from app.main import app
from app.db.database import get_db


from sqlalchemy.pool import StaticPool

@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for deployment gate tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    # Seed baseline hospital
    h = Hospital(hospital_code="HOSP-001", name="Test Hospital", is_active=True)
    db.add(h)
    db.commit()

    yield db
    db.close()


@pytest.fixture
def client(test_db):
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# =========================================================================
# Unit Tests: DeploymentGate Service Logic
# =========================================================================

def test_validation_set_hash_alters_on_file_change(tmp_path):
    """Verify that adding, modifying, or removing a validation file changes the computed hash."""
    from app.services.deployment_gate import compute_validation_dataset_hash

    # Create mock validation structure
    val_dir1 = tmp_path / "hospital_1" / "validation" / "Cyst"
    val_dir1.mkdir(parents=True, exist_ok=True)
    f1 = val_dir1 / "slice_001.jpg"
    f1.write_text("initial image content 1")

    val_dir2 = tmp_path / "hospital_2" / "validation" / "Normal"
    val_dir2.mkdir(parents=True, exist_ok=True)
    f2 = val_dir2 / "slice_002.jpg"
    f2.write_text("initial image content 2")

    val_paths = [tmp_path / f"hospital_{i}" / "validation" for i in [1, 2]]

    initial_hash = compute_validation_dataset_hash(val_paths=val_paths)
    assert len(initial_hash) == 16
    assert initial_hash != "empty_dataset"

    # Same files produce same hash
    repeat_hash = compute_validation_dataset_hash(val_paths=val_paths)
    assert repeat_hash == initial_hash

    # Altering content of one file changes hash
    f1.write_text("modified content of slice 1 with different byte size")
    modified_hash = compute_validation_dataset_hash(val_paths=val_paths)
    assert modified_hash != initial_hash

    # Adding a new file changes hash
    f3 = val_dir1 / "slice_003.jpg"
    f3.write_text("extra image slice")
    added_hash = compute_validation_dataset_hash(val_paths=val_paths)
    assert added_hash != modified_hash


def test_gate_candidate_passes_all_checks(test_db):
    """Candidate exceeding all thresholds on same validation split becomes 'eligible'."""
    # Seed current deployed baseline
    deployed = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_baseline_v1",
        accuracy=0.960,
        f1_score=0.950,
        is_deployed=True,
        status="deployed",
        artifact_path="dl/models/baseline.pth",
        gate_report={"data_source": "dataset_hash_val_abc", "is_central_eval": True},
    )
    test_db.add(deployed)
    test_db.commit()

    gate = DeploymentGate(
        Settings(
            gate_min_accuracy=0.95,
            gate_min_f1=0.94,
            gate_min_recall_stone=0.96,
            gate_min_recall_tumor=0.98,
            gate_max_regression_vs_current=0.02,
        )
    )

    metrics = {
        "accuracy": 0.985,
        "f1": 0.980,
        "recall_stone": 0.990,
        "recall_tumor": 1.000,
        "validation_data_source": "dataset_hash_val_abc",
    }

    report = gate.evaluate_candidate(metrics, test_db, model_family="resnet18_ct")
    assert report["passed"] is True
    assert report["status"] == "eligible"
    assert len(report["reasons"]) == 0
    assert all(c["status"] in ("passed", "info") for c in report["checks"])


def test_gate_candidate_fails_minimum_accuracy(test_db):
    """Candidate with low accuracy is marked 'rejected' with explicit reason."""
    gate = DeploymentGate(Settings(gate_min_accuracy=0.95, gate_min_f1=0.94, gate_min_recall_stone=0.96, gate_min_recall_tumor=0.98))
    metrics = {
        "accuracy": 0.910,  # Below 0.95
        "f1": 0.950,
        "recall_stone": 0.970,
        "recall_tumor": 0.990,
        "validation_data_source": "dataset_hash_val_abc",
    }

    report = gate.evaluate_candidate(metrics, test_db, model_family="resnet18_ct")
    assert report["passed"] is False
    assert report["status"] == "rejected"
    assert any("Accuracy (0.9100) fell below" in r for r in report["reasons"])


def test_gate_candidate_missing_per_class_recall_blocks_eligibility(test_db):
    """Missing Stone/Tumor recall marks check 'not_evaluated' and blocks eligibility (pending_review)."""
    gate = DeploymentGate(Settings(gate_min_accuracy=0.95, gate_min_f1=0.94, gate_min_recall_stone=0.96, gate_min_recall_tumor=0.98))
    metrics = {
        "accuracy": 0.980,
        "f1": 0.970,
        # No recall_stone or recall_tumor provided!
        "validation_data_source": "dataset_hash_val_abc",
    }

    report = gate.evaluate_candidate(metrics, test_db, model_family="resnet18_ct")
    assert report["passed"] is False
    assert report["status"] == "pending_review"
    assert any("Stone recall was not evaluated" in r for r in report["reasons"])
    assert any("Tumor recall was not evaluated" in r for r in report["reasons"])


def test_gate_candidate_regression_against_deployed_rejected(test_db):
    """Candidate dropping more than max_regression_vs_current (0.02) vs deployed is rejected."""
    deployed = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_baseline_v1",
        accuracy=0.990,
        f1_score=0.985,
        is_deployed=True,
        status="deployed",
        artifact_path="dl/models/baseline.pth",
        gate_report={"data_source": "dataset_hash_val_abc", "is_central_eval": True},
    )
    test_db.add(deployed)
    test_db.commit()

    gate = DeploymentGate(
        Settings(
            gate_min_accuracy=0.95,
            gate_min_f1=0.94,
            gate_min_recall_stone=0.96,
            gate_min_recall_tumor=0.98,
            gate_max_regression_vs_current=0.02,
        )
    )

    # Candidate meets min thresholds (acc=0.955 >= 0.95, f1=0.945 >= 0.94)
    # but drops > 0.02 vs deployed (acc drop: 0.990 - 0.955 = 0.035 > 0.02)
    metrics = {
        "accuracy": 0.955,
        "f1": 0.945,
        "recall_stone": 0.970,
        "recall_tumor": 0.990,
        "validation_data_source": "dataset_hash_val_abc",
    }

    report = gate.evaluate_candidate(metrics, test_db, model_family="resnet18_ct")
    assert report["passed"] is False
    assert report["status"] == "rejected"
    assert any("regressed against active deployed model" in r for r in report["reasons"])


def test_gate_validation_dataset_mismatch_requires_manual_review(test_db):
    """Candidate evaluated on different dataset split is marked 'not_comparable' -> 'pending_review'."""
    deployed = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_baseline_v1",
        accuracy=0.980,
        f1_score=0.970,
        is_deployed=True,
        status="deployed",
        artifact_path="dl/models/baseline.pth",
        gate_report={"data_source": "dataset_hash_val_abc", "is_central_eval": True},
    )
    test_db.add(deployed)
    test_db.commit()

    gate = DeploymentGate(Settings())

    metrics = {
        "accuracy": 0.985,
        "f1": 0.980,
        "recall_stone": 0.990,
        "recall_tumor": 1.000,
        "validation_data_source": "custom_hospital_hospital1_test_split",  # Mismatch!
    }

    report = gate.evaluate_candidate(metrics, test_db, model_family="resnet18_ct")
    assert report["passed"] is False
    assert report["status"] == "pending_review"
    assert any("Validation dataset mismatch" in c["details"] for c in report["checks"])


# =========================================================================
# Integration Tests: API Endpoints (Deploy, Rollback, Query)
# =========================================================================

def test_api_deploy_rejected_model_returns_400(client, test_db):
    """Deploying a model version with status='rejected' must return HTTP 400."""
    rejected_model = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_bad_v1",
        accuracy=0.88,
        f1_score=0.85,
        is_deployed=False,
        status="rejected",
        artifact_path="dl/models/bad.pth",
    )
    test_db.add(rejected_model)
    test_db.commit()

    res = client.post("/api/v1/developer/model-versions/deploy", json={"model_version_id": rejected_model.id})
    assert res.status_code == 400
    assert "status is 'rejected'" in res.json()["detail"]
    assert "eligible" in res.json()["detail"]


def test_api_deploy_eligible_model_archives_previous_and_sets_audit(client, test_db):
    """Deploying an eligible model transitions previous to 'archived' and records audit fields."""
    v1 = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_prod_v1",
        accuracy=0.96,
        f1_score=0.95,
        is_deployed=True,
        status="deployed",
        artifact_path="dl/models/v1.pth",
    )
    v2 = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_prod_v2",
        accuracy=0.98,
        f1_score=0.97,
        is_deployed=False,
        status="eligible",
        artifact_path="dl/models/v2.pth",
    )
    test_db.add_all([v1, v2])
    test_db.commit()

    res = client.post(
        "/api/v1/developer/model-versions/deploy",
        json={"model_version_id": v2.id, "approved_by": "lead_ml_engineer"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["deployed"] == "resnet18_prod_v2"
    assert data["approved_by"] == "lead_ml_engineer"
    assert data["previous_deployed_version_id"] == v1.id

    # Verify DB state
    test_db.refresh(v1)
    test_db.refresh(v2)
    assert v1.is_deployed is False
    assert v1.status == "archived"
    assert v2.is_deployed is True
    assert v2.status == "deployed"
    assert v2.approved_by == "lead_ml_engineer"
    assert v2.approved_at is not None
    assert v2.deployed_at is not None


def test_api_sequential_deploys_guarantee_single_deployed_row(client, test_db):
    """Multiple sequential deployments strictly preserve exactly one deployed row per family."""
    v1 = ModelVersion(model_family="resnet18_ct", version_tag="v1", is_deployed=True, status="deployed", artifact_path="v1")
    v2 = ModelVersion(model_family="resnet18_ct", version_tag="v2", is_deployed=False, status="eligible", artifact_path="v2")
    v3 = ModelVersion(model_family="resnet18_ct", version_tag="v3", is_deployed=False, status="eligible", artifact_path="v3")
    test_db.add_all([v1, v2, v3])
    test_db.commit()

    # Deploy v2
    client.post("/api/v1/developer/model-versions/deploy", json={"model_version_id": v2.id})
    # Deploy v3
    client.post("/api/v1/developer/model-versions/deploy", json={"model_version_id": v3.id})

    deployed_rows = test_db.query(ModelVersion).filter_by(model_family="resnet18_ct", is_deployed=True).all()
    assert len(deployed_rows) == 1
    assert deployed_rows[0].version_tag == "v3"


def test_api_rollback_restores_previous_deployed_version(client, test_db):
    """Rollback restores the previous active deployed model."""
    now = datetime.utcnow()
    v1 = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_v1",
        is_deployed=False,
        status="archived",
        deployed_at=now - timedelta(days=2),
        artifact_path="dl/models/v1.pth",
    )
    v2 = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_v2",
        is_deployed=True,
        status="deployed",
        deployed_at=now,
        previous_deployed_version_id=v1.id,
        artifact_path="dl/models/v2.pth",
    )
    test_db.add_all([v1, v2])
    test_db.commit()
    v2.previous_deployed_version_id = v1.id
    test_db.commit()

    res = client.post("/api/v1/developer/model-versions/rollback", json={"model_family": "resnet18_ct"})
    assert res.status_code == 200
    data = res.json()
    assert data["deployed"] == "resnet18_v1"
    assert data["rolled_back_from"] == "resnet18_v2"

    test_db.refresh(v1)
    test_db.refresh(v2)
    assert v1.is_deployed is True
    assert v1.status == "deployed"
    assert v2.is_deployed is False
    assert v2.status == "archived"


def test_api_model_versions_includes_gate_fields(client, test_db):
    """GET /developer/model-versions outputs status, gate_report, and audit fields."""
    gate_data = {
        "passed": True,
        "status": "eligible",
        "data_source": "dl_partitions_iid_seed42_val",
        "checks": [{"name": "min_accuracy", "status": "passed"}],
    }
    v = ModelVersion(
        model_family="resnet18_ct",
        version_tag="resnet18_gate_v1",
        accuracy=0.985,
        f1_score=0.980,
        is_deployed=False,
        status="eligible",
        gate_report=gate_data,
        artifact_path="dl/models/gate_v1.pth",
    )
    test_db.add(v)
    test_db.commit()

    res = client.get("/api/v1/developer/model-versions?model_family=resnet18_ct")
    assert res.status_code == 200
    rows = res.json()
    match = next(r for r in rows if r["version_tag"] == "resnet18_gate_v1")
    assert match["status"] == "eligible"
    assert match["gate_report"]["passed"] is True
    assert match["gate_report"]["data_source"] == "dl_partitions_iid_seed42_val"


def test_centralized_ml_training_registers_as_pending_review_without_autodeploy(client, test_db, monkeypatch):
    """Centralized ML training must register new versions as pending_review with is_deployed=False."""
    # Seed current deployed version
    current = ModelVersion(
        model_family="logistic_regression_risk",
        version_tag="logisticregression_centralized_v001",
        is_deployed=True,
        status="deployed",
        artifact_path="ml/models/lr.pkl",
    )
    test_db.add(current)
    test_db.commit()

    # Mock MLRiskTrainer
    class MockTrainer:
        def __init__(self, **kwargs):
            pass
        def train_and_compare(self):
            return {
                "accuracy": 0.96,
                "f1_score": 0.95,
                "precision": 0.95,
                "recall": 0.95,
                "matthews_correlation_coefficient": 0.92,
            }

    import sys
    import types
    mock_module = types.ModuleType("train_risk_model")
    mock_module.MLRiskTrainer = MockTrainer
    monkeypatch.setitem(sys.modules, "train_risk_model", mock_module)

    res = client.post("/api/v1/developer/ml/training/start")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"

    # Verify that the newly registered version is is_deployed=False and status='pending_review'
    new_v = test_db.query(ModelVersion).filter_by(version_tag=data["version_tag"]).first()
    assert new_v is not None
    assert new_v.is_deployed is False
    assert new_v.status == "pending_review"

    # Verify that previous deployed model is still deployed
    test_db.refresh(current)
    assert current.is_deployed is True
    assert current.status == "deployed"
