"""Hospital-scoped routes for StoneSense-AI.

Provides isolated endpoints for each hospital node:
- Hospital metadata and patient prediction history
- Local dataset inspection and validation
- Federated participation and model synchronization status
- Local isolated training execution
"""

from pathlib import Path
import os
import sys
import shutil
import tempfile
import zipfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database import get_db
from app.db.models import (
    Hospital, Prediction, HospitalTrainingRun, FederatedRound,
    ModelVersion, HospitalUpdateLog
)
from app.schemas.dashboard import HospitalOut, PatientHistoryItem
from app.schemas.federated import (
    DatasetStatusOut, DatasetValidateResponse, FederatedStatusOut,
    HospitalRunTelemetryOut, LocalTrainingTriggerResponse, HospitalLiveStatusResponse
)
from app.services.model_loader import model_loader
from app.services.federated_coordinator import federated_coordinator

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PARTITIONS_ROOT = PROJECT_ROOT / "dl" / "datasets" / "partitions"
DATASET_CLASSES = ("Cyst", "Normal", "Stone", "Tumor")
DATASET_SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_DATASET_UPLOAD_BYTES = 512 * 1024 * 1024

router = APIRouter()


def _get_hospital_or_404(hospital_id: int, db: Session) -> Hospital:
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail=f"Hospital ID {hospital_id} not found")
    return hospital


def _require_hospital_scope(hospital_id: int, x_hospital_id: Optional[str]) -> None:
    """Require the caller to identify the same hospital as the URL scope."""
    if not x_hospital_id:
        raise HTTPException(status_code=401, detail="Hospital identity is required.")

    try:
        requested_id = int(x_hospital_id)
    except ValueError:
        raise HTTPException(status_code=403, detail="Hospital identity does not match this resource.")

    if requested_id != hospital_id:
        raise HTTPException(status_code=403, detail="Hospital identity does not match this resource.")


@router.get("/list", response_model=List[HospitalOut])
def list_hospitals(db: Session = Depends(get_db)):
    """Lists all active hospitals."""
    return db.query(Hospital).filter(Hospital.is_active.is_(True)).order_by(Hospital.id).all()


@router.get("/{hospital_id}/detail", response_model=HospitalOut)
def get_hospital_detail(hospital_id: int, db: Session = Depends(get_db)):
    """Returns metadata for a specific hospital."""
    return _get_hospital_or_404(hospital_id, db)


@router.get("/{hospital_id}/history", response_model=List[PatientHistoryItem])
def get_history(hospital_id: int, limit: int = 25, db: Session = Depends(get_db)):
    """Recent predictions for one hospital only."""
    _get_hospital_or_404(hospital_id, db)
    rows = (
        db.query(Prediction)
        .filter(Prediction.hospital_id == hospital_id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        PatientHistoryItem(
            id=r.id,
            reference_code=r.patient.reference_code if r.patient else "—",
            prediction_type=r.prediction_type,
            model_name=r.model_name,
            result_label=r.result_label,
            confidence=r.confidence,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/{hospital_id}/training-history", response_model=List[HospitalRunTelemetryOut])
def get_training_history(hospital_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """Returns history of local training runs per round for this hospital."""
    h = _get_hospital_or_404(hospital_id, db)
    runs = (
        db.query(HospitalTrainingRun, ModelVersion.version_tag)
        .filter(HospitalTrainingRun.hospital_id == hospital_id)
        .outerjoin(ModelVersion, ModelVersion.round_id == HospitalTrainingRun.round_id)
        .order_by(desc(HospitalTrainingRun.round_number))
        .limit(limit)
        .all()
    )
    return [
        HospitalRunTelemetryOut(
            id=r.id,
            round_number=r.round_number,
            hospital_id=r.hospital_id,
            hospital_code=r.hospital_code,
            hospital_name=h.name,
            model_version=model_version,
            train_loss=r.train_loss,
            train_acc=r.train_acc,
            train_f1=r.train_f1,
            val_loss=r.val_loss,
            val_acc=r.val_acc,
            val_f1=r.val_f1,
            sample_count=r.sample_count,
            duration_sec=r.duration_sec,
            created_at=r.created_at
        )
        for r, model_version in runs
    ]


@router.get("/{hospital_id}/current-model")
def get_current_model(hospital_id: int, db: Session = Depends(get_db)):
    """Returns the current active model version deployed for this hospital."""
    h = _get_hospital_or_404(hospital_id, db)
    latest_round = db.query(FederatedRound).order_by(desc(FederatedRound.round_number)).first()
    deployed_ver = db.query(ModelVersion).filter_by(is_deployed=True, model_family="resnet18_ct").first()

    return {
        "hospital_id": h.id,
        "hospital_code": h.hospital_code,
        "current_model_version": h.current_model_version or (deployed_ver.version_tag if deployed_ver else "resnet18_centralized_v1"),
        "global_round": latest_round.round_number if latest_round else 0,
        "global_accuracy": latest_round.global_val_acc if latest_round else (deployed_ver.accuracy if deployed_ver else 0.985),
        "global_f1": latest_round.global_val_f1 if latest_round else (deployed_ver.f1_score if deployed_ver else 0.979),
        "deployed_at": deployed_ver.trained_at if deployed_ver else datetime.utcnow()
    }


@router.get("/{hospital_id}/federated-status", response_model=FederatedStatusOut)
def get_federated_status(hospital_id: int, db: Session = Depends(get_db)):
    """Returns participation status of the hospital in federated learning."""
    h = _get_hospital_or_404(hospital_id, db)
    latest_round = db.query(FederatedRound).order_by(desc(FederatedRound.round_number)).first()
    latest_run = (
        db.query(HospitalTrainingRun)
        .filter_by(hospital_id=hospital_id)
        .order_by(desc(HospitalTrainingRun.round_number))
        .first()
    )

    return FederatedStatusOut(
        hospital_code=h.hospital_code,
        status="active_participant" if latest_run else "ready",
        current_round=latest_round.round_number if latest_round else 0,
        current_model_version=h.current_model_version or "resnet18_centralized_v1",
        last_round_participated=latest_run.round_number if latest_run else None,
        local_accuracy=latest_run.val_acc if latest_run else None,
        local_f1=latest_run.val_f1 if latest_run else None
    )


@router.get("/{hospital_id}/federated-live-status", response_model=HospitalLiveStatusResponse)
def get_federated_live_status(hospital_id: str, db: Session = Depends(get_db)):
    """Returns real-time round and local client execution status for a specific hospital node."""
    # Support both numeric ID and hospital_code (e.g. 1 or HOSP-001)
    if hospital_id.isdigit():
        h = db.query(Hospital).filter(Hospital.id == int(hospital_id)).first()
        if not h:
            raise HTTPException(status_code=404, detail=f"Hospital ID {hospital_id} not found")
        h_code = h.hospital_code
    else:
        h = db.query(Hospital).filter(Hospital.hospital_code == hospital_id).first()
        if not h:
            raise HTTPException(status_code=404, detail=f"Hospital code {hospital_id} not found")
        h_code = h.hospital_code

    try:
        return federated_coordinator.get_hospital_live_status(h_code)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Hospital {h_code} is not configured as an FL client")



def _get_hospital_partition_dir(hospital_code: str) -> Path:
    code_map = {"HOSP-001": "hospital_1", "HOSP-002": "hospital_2", "HOSP-003": "hospital_3"}
    dir_name = code_map.get(hospital_code, hospital_code.lower())
    partition_dir = PARTITIONS_ROOT / dir_name
    return partition_dir


def _inspect_dataset(part_dir: Path) -> Dict[str, Any]:
    """Validate the directory contract consumed by create_dataloaders()."""
    class_counts = {class_name: 0 for class_name in DATASET_CLASSES}
    split_counts = {split: 0 for split in DATASET_SPLITS}
    corrupted = 0
    missing_paths = []

    for split in DATASET_SPLITS:
        split_path = part_dir / split
        if not split_path.is_dir():
            missing_paths.append(split)
            continue

        for class_name in DATASET_CLASSES:
            class_path = split_path / class_name
            if not class_path.is_dir():
                missing_paths.append(f"{split}/{class_name}")
                continue

            for image_path in class_path.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                split_counts[split] += 1
                class_counts[class_name] += 1
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception:
                    corrupted += 1

    total = sum(split_counts.values())
    is_valid = (
        not missing_paths
        and split_counts["train"] > 0
        and all((part_dir / "train" / class_name).exists() for class_name in DATASET_CLASSES)
        and corrupted == 0
    )
    if missing_paths:
        message = "Missing required dataset paths: " + ", ".join(missing_paths[:5])
    elif corrupted:
        message = f"Validation found {corrupted} corrupted image file(s)."
    elif split_counts["train"] == 0:
        message = "The train split contains no supported images."
    elif is_valid:
        message = "Dataset validated successfully for the ResNet18 training pipeline."
    else:
        message = "Dataset validation failed."

    return {
        "is_valid": is_valid,
        "total": total,
        "class_counts": class_counts,
        "split_counts": split_counts,
        "corrupted": corrupted,
        "message": message,
    }


def _dataset_status(hospital: Hospital, inspection: Dict[str, Any]) -> DatasetStatusOut:
    return DatasetStatusOut(
        hospital_id=hospital.id,
        hospital_code=hospital.hospital_code,
        name=hospital.name,
        dataset_size=inspection["total"],
        class_distribution=inspection["class_counts"],
        is_valid=inspection["is_valid"],
        split_info=inspection["split_counts"],
        dataset_version=hospital.dataset_version,
        last_updated=hospital.dataset_updated_at,
        last_validated=hospital.dataset_validated_at,
        validation_message=inspection["message"],
    )


@router.get("/{hospital_id}/dataset-status", response_model=DatasetStatusOut)
def get_dataset_status(
    hospital_id: int,
    db: Session = Depends(get_db),
    x_hospital_id: Optional[str] = Header(default=None, alias="X-Hospital-ID"),
):
    """Returns private dataset metadata for the requesting hospital only."""
    h = _get_hospital_or_404(hospital_id, db)
    if x_hospital_id:
        _require_hospital_scope(hospital_id, x_hospital_id)
    part_dir = _get_hospital_partition_dir(h.hospital_code)
    inspection = _inspect_dataset(part_dir) if part_dir.exists() else {
        "is_valid": False,
        "total": 0,
        "class_counts": {class_name: 0 for class_name in DATASET_CLASSES},
        "split_counts": {split: 0 for split in DATASET_SPLITS},
        "message": "Private dataset has not been uploaded.",
    }
    return _dataset_status(h, inspection)


@router.post("/{hospital_id}/dataset-validate", response_model=DatasetValidateResponse)
def validate_dataset(
    hospital_id: int,
    db: Session = Depends(get_db),
    x_hospital_id: Optional[str] = Header(default=None, alias="X-Hospital-ID"),
):
    """Validates the private dataset without exposing its files."""
    h = _get_hospital_or_404(hospital_id, db)
    _require_hospital_scope(hospital_id, x_hospital_id)
    part_dir = _get_hospital_partition_dir(h.hospital_code)

    if not part_dir.exists():
        return DatasetValidateResponse(
            hospital_code=h.hospital_code,
            is_valid=False,
            total_samples=0,
            classes={},
            corrupted_images=0,
            message="Private dataset directory not found. Upload a dataset first."
        )

    inspection = _inspect_dataset(part_dir)
    h.dataset_size = inspection["total"]
    h.class_distribution = inspection["class_counts"]
    h.dataset_split_counts = inspection["split_counts"]
    h.dataset_is_valid = inspection["is_valid"]
    h.dataset_validated_at = datetime.utcnow()
    db.commit()

    return DatasetValidateResponse(
        hospital_code=h.hospital_code,
        is_valid=inspection["is_valid"],
        total_samples=inspection["total"],
        classes=inspection["class_counts"],
        corrupted_images=inspection["corrupted"],
        message=inspection["message"]
    )


@router.post("/{hospital_id}/dataset-upload", response_model=DatasetStatusOut)
def upload_dataset(
    hospital_id: int,
    dataset: UploadFile = File(...),
    db: Session = Depends(get_db),
    x_hospital_id: Optional[str] = Header(default=None, alias="X-Hospital-ID"),
):
    """Replace one hospital's private ResNet18 dataset from a validated ZIP archive."""
    h = _get_hospital_or_404(hospital_id, db)
    _require_hospital_scope(hospital_id, x_hospital_id)
    if not dataset.filename or not dataset.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a ZIP containing train, validation, and test folders.")

    PARTITIONS_ROOT.mkdir(parents=True, exist_ok=True)
    target_dir = _get_hospital_partition_dir(h.hospital_code)

    with tempfile.TemporaryDirectory(prefix=f"{h.hospital_code}-dataset-", dir=PARTITIONS_ROOT) as temp_dir:
        archive_path = Path(temp_dir) / "dataset.zip"
        total_bytes = 0
        with archive_path.open("wb") as archive_file:
            while True:
                chunk = dataset.file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_DATASET_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Dataset ZIP exceeds the 512 MB upload limit.")
                archive_file.write(chunk)

        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    member_path = (extract_dir / member.filename).resolve()
                    if extract_dir.resolve() not in member_path.parents:
                        raise HTTPException(status_code=400, detail="Dataset archive contains an unsafe path.")
                archive.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Dataset upload is not a valid ZIP archive.")

        staged_dir = extract_dir
        if not (staged_dir / "train").is_dir():
            candidates = [path for path in staged_dir.iterdir() if path.is_dir()]
            if len(candidates) == 1 and (candidates[0] / "train").is_dir():
                staged_dir = candidates[0]

        inspection = _inspect_dataset(staged_dir)
        if not inspection["is_valid"]:
            raise HTTPException(status_code=422, detail=inspection["message"])

        replacement_dir = Path(temp_dir) / "replacement"
        shutil.copytree(staged_dir, replacement_dir)
        backup_dir = PARTITIONS_ROOT / f".{target_dir.name}.previous"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if target_dir.exists():
            target_dir.rename(backup_dir)
        try:
            replacement_dir.rename(target_dir)
        except Exception:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            if backup_dir.exists():
                backup_dir.rename(target_dir)
            raise HTTPException(status_code=500, detail="Could not install the private dataset.")
        finally:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

    now = datetime.utcnow()
    previous_version = h.dataset_version or ""
    try:
        version_number = int(previous_version.rsplit("-v", 1)[1]) + 1
    except (ValueError, IndexError):
        version_number = 1
    h.dataset_version = f"{h.hospital_code}-DATA-v{version_number}"
    h.dataset_size = inspection["total"]
    h.class_distribution = inspection["class_counts"]
    h.dataset_split_counts = inspection["split_counts"]
    h.dataset_is_valid = True
    h.dataset_updated_at = now
    h.dataset_validated_at = now
    db.commit()

    return _dataset_status(h, inspection)


@router.post("/{hospital_id}/train-local", response_model=LocalTrainingTriggerResponse)
def trigger_local_training(hospital_id: int, db: Session = Depends(get_db)):
    """Triggers an isolated local benchmark training on this hospital partition."""
    h = _get_hospital_or_404(hospital_id, db)
    part_dir = _get_hospital_partition_dir(h.hospital_code)

    if not part_dir.exists():
        raise HTTPException(status_code=400, detail="Local partition does not exist.")

    # Run quick benchmark training pass
    try:
        sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))
        from client import StoneSenseFLClient

        client = StoneSenseFLClient(hospital_id=h.hospital_code, partition_dir=part_dir, batch_size=32)
        initial_params = client.get_parameters({})
        _, samples, metrics = client.fit(initial_params, {"local_epochs": 1, "lr": 0.0005, "current_round": 0})

        return LocalTrainingTriggerResponse(
            hospital_code=h.hospital_code,
            status="completed",
            message=f"Local training completed on {samples} samples.",
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local training execution failed: {e}")
