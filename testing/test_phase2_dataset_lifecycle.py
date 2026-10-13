"""Phase 2 End-to-End Dataset Lifecycle, Candidate Review & Deployment Gate Tests.

Validates:
1. Local partition manifest generation, SHA-256 hashing, duplicate detection.
2. Patient/scan-level grouping integrity and cross-split leakage detection.
3. Delta change scanning (no retrain if no new valid data).
4. Corrupted file detection and integrity rejection.
5. Hospital dataset validation API integration.
6. Local training skip behavior when require_new_data is True and no new scans exist.
7. Candidate review workflow (explicit approve / reject) and deployment gating (only eligible deployed).
8. Model deployment and rollback consistency.
"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT))

from app.main import app
from app.db.database import init_db, SessionLocal
from app.db.models import ModelVersion, Hospital, User
from app.core.security import create_access_token
from dl.federated.dataset_manifest import (
    scan_partition_dataset, compute_file_sha256, extract_patient_group,
    DATASET_CLASSES, DATASET_SPLITS
)

dev_token = create_access_token(data={"sub": "dev@stonesense.ai", "role": "developer", "hospital_id": None})
hosp1_token = create_access_token(data={"sub": "hospital1@stonesense.ai", "role": "hospital_user", "hospital_id": 1})

dev_client = TestClient(app, headers={"Authorization": f"Bearer {dev_token}"})
hosp_client = TestClient(app, headers={"Authorization": f"Bearer {hosp1_token}"})


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def create_dummy_image(path: Path, color=(200, 200, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


def test_extract_patient_group():
    """Verifies clinical patient identifier extraction across diverse naming conventions."""
    assert extract_patient_group("patient_0012_slice_05.png") == "patient_0012"
    assert extract_patient_group("case_45_scan_1.jpg") == "case_45"
    assert extract_patient_group("p123_45.bmp") == "p123"
    assert extract_patient_group("CT_009_slice02.png") == "ct_009"


def test_manifest_clean_dataset_and_delta():
    """Verifies SHA-256 calculation, manifest generation, and delta tracking."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        part_dir = Path(tmp_dir)
        # Create standard structure
        for split in DATASET_SPLITS:
            for cls in DATASET_CLASSES:
                create_dummy_image(part_dir / split / cls / f"{split}_{cls}_case01_slice1.png")

        # First scan
        manifest1 = scan_partition_dataset(part_dir, hospital_code="HOSP-TEST", save_manifest=True)
        assert manifest1["corrupted_count"] == 0
        assert manifest1["patient_leakage_detected"] is False
        assert manifest1["total_images"] == len(DATASET_SPLITS) * len(DATASET_CLASSES)
        assert manifest1["has_new_valid_training_data"] is True
        assert (part_dir / "manifest.json").exists()

        # Second scan without changes -> has_new_valid_training_data should be False
        manifest2 = scan_partition_dataset(part_dir, hospital_code="HOSP-TEST", save_manifest=True)
        assert manifest2["delta"]["added_count"] == 0
        assert manifest2["delta"]["new_train_images"] == 0
        assert manifest2["has_new_valid_training_data"] is False


def test_manifest_detects_corrupted_images():
    """Verifies corrupted file detection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        part_dir = Path(tmp_dir)
        # Create one valid, one corrupted image
        valid_path = part_dir / "train" / "Stone" / "case1_slice1.png"
        corrupt_path = part_dir / "train" / "Stone" / "case1_slice2.png"
        create_dummy_image(valid_path)
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corrupt_path, "wb") as f:
            f.write(b"not an image corrupted data")

        manifest = scan_partition_dataset(part_dir, hospital_code="HOSP-TEST", save_manifest=False)
        assert manifest["corrupted_count"] == 1
        assert manifest["has_new_valid_training_data"] is False


def test_manifest_detects_patient_leakage_across_splits():
    """Verifies detection of patient ID leakage between train and validation splits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        part_dir = Path(tmp_dir)
        # Leaking patient 'patient_99' in train and validation
        create_dummy_image(part_dir / "train" / "Stone" / "patient_99_slice1.png")
        create_dummy_image(part_dir / "validation" / "Stone" / "patient_99_slice2.png")
        create_dummy_image(part_dir / "test" / "Stone" / "patient_88_slice1.png")

        manifest = scan_partition_dataset(part_dir, hospital_code="HOSP-TEST", save_manifest=False)
        assert manifest["patient_leakage_detected"] is True
        assert manifest["leaked_patient_count"] >= 1
        assert manifest["has_new_valid_training_data"] is False


def test_hospital_dataset_validate_endpoint():
    """Verifies POST /{hospital_id}/dataset-validate integrates manifest scanning."""
    res = hosp_client.post("/api/v1/hospital/1/dataset-validate")
    assert res.status_code == 200
    data = res.json()
    assert data["hospital_code"] == "HOSP-001"
    assert "is_valid" in data
    assert "total_samples" in data
    assert "classes" in data
    assert "duplicate_count" in data
    assert "patient_leakage_detected" in data


def test_hospital_local_training_skip_without_new_data():
    """Verifies that require_new_data=true skips training when no new valid training images exist."""
    # Ensure manifest is up-to-date
    hosp_client.post("/api/v1/hospital/1/dataset-validate")
    # Trigger local training with require_new_data=true
    res = hosp_client.post("/api/v1/hospital/1/train-local?require_new_data=true")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "skipped"
    assert "skipped" in data["message"].lower()
    assert data["samples_used"] == 0


def test_candidate_review_approve_deploy_and_rollback():
    """Verifies the explicit review workflow: approve/reject, gate enforcement, deploy, and rollback."""
    db = SessionLocal()
    # Create test model version
    tag_v1 = f"test_resnet_v1_{int(datetime.utcnow().timestamp())}"
    tag_v2 = f"test_resnet_v2_{int(datetime.utcnow().timestamp())}"

    mv1 = ModelVersion(
        model_family="resnet18_ct",
        version_tag=tag_v1,
        accuracy=0.95,
        f1_score=0.94,
        is_deployed=True,
        status="deployed",
        artifact_path="/dummy/path1",
        deployed_at=datetime.utcnow(),
    )
    mv2 = ModelVersion(
        model_family="resnet18_ct",
        version_tag=tag_v2,
        accuracy=0.97,
        f1_score=0.96,
        is_deployed=False,
        status="pending_review",
        gate_report={"status": "eligible", "reasons": []},
        artifact_path="/dummy/path2",
    )
    db.add(mv1)
    db.add(mv2)
    db.commit()
    mv1_id, mv2_id = mv1.id, mv2.id
    db.close()

    try:
        # 1. Rejecting mv2
        res_reject = dev_client.post(
            "/api/v1/developer/model-versions/review",
            json={"model_version_id": mv2_id, "decision": "reject", "notes": "Clinical review rejected"}
        )
        assert res_reject.status_code == 200
        assert res_reject.json()["status"] == "rejected"

        # Attempting to deploy rejected model must fail with 400
        res_deploy_fail = dev_client.post(
            "/api/v1/developer/model-versions/deploy",
            json={"model_version_id": mv2_id}
        )
        assert res_deploy_fail.status_code == 400

        # 2. Approving mv2
        res_approve = dev_client.post(
            "/api/v1/developer/model-versions/review",
            json={"model_version_id": mv2_id, "decision": "approve", "notes": "Approved for clinical use"}
        )
        assert res_approve.status_code == 200
        assert res_approve.json()["status"] == "eligible"

        # 3. Deploying approved mv2 succeeds
        res_deploy = dev_client.post(
            "/api/v1/developer/model-versions/deploy",
            json={"model_version_id": mv2_id}
        )
        assert res_deploy.status_code == 200
        assert res_deploy.json()["status"] == "deployed"

        db_check = SessionLocal()
        v2_db = db_check.query(ModelVersion).filter_by(id=mv2_id).first()
        v1_db = db_check.query(ModelVersion).filter_by(id=mv1_id).first()
        assert v2_db.is_deployed is True
        assert v1_db.is_deployed is False
        db_check.close()

        # 4. Rollback promotes previous version (mv1) back to deployed
        res_rollback = dev_client.post(
            "/api/v1/developer/model-versions/rollback",
            json={"model_family": "resnet18_ct", "target_version_id": mv1_id}
        )
        assert res_rollback.status_code == 200
        assert res_rollback.json()["deployed"] == tag_v1

        db_check2 = SessionLocal()
        v1_rolled = db_check2.query(ModelVersion).filter_by(id=mv1_id).first()
        v2_rolled = db_check2.query(ModelVersion).filter_by(id=mv2_id).first()
        assert v1_rolled.is_deployed is True
        assert v2_rolled.is_deployed is False
        db_check2.close()

    finally:
        # Cleanup
        db_clean = SessionLocal()
        db_clean.query(ModelVersion).filter(ModelVersion.id.in_([mv1_id, mv2_id])).delete(synchronize_session=False)
        db_clean.commit()
        db_clean.close()
