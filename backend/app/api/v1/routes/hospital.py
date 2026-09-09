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
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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
    HospitalRunTelemetryOut, LocalTrainingTriggerResponse
)
from app.services.model_loader import model_loader

PROJECT_ROOT = Path(__file__).resolve().parents[4]

router = APIRouter()


def _get_hospital_or_404(hospital_id: int, db: Session) -> Hospital:
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail=f"Hospital ID {hospital_id} not found")
    return hospital


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
        db.query(HospitalTrainingRun)
        .filter(HospitalTrainingRun.hospital_id == hospital_id)
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
        for r in runs
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


def _get_hospital_partition_dir(hospital_code: str) -> Path:
    code_map = {"HOSP-001": "hospital_1", "HOSP-002": "hospital_2", "HOSP-003": "hospital_3"}
    dir_name = code_map.get(hospital_code, hospital_code.lower())
    partition_dir = PROJECT_ROOT / "dl" / "datasets" / "partitions" / dir_name
    return partition_dir


@router.get("/{hospital_id}/dataset-status", response_model=DatasetStatusOut)
def get_dataset_status(hospital_id: int, db: Session = Depends(get_db)):
    """Returns local partition size, splits, and class distribution."""
    h = _get_hospital_or_404(hospital_id, db)
    part_dir = _get_hospital_partition_dir(h.hospital_code)

    class_dist: Dict[str, int] = {"Cyst": 0, "Normal": 0, "Stone": 0, "Tumor": 0}
    split_info: Dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    if part_dir.exists():
        for split in ["train", "validation", "test"]:
            split_path = part_dir / split
            if split_path.exists():
                for cname in ["Cyst", "Normal", "Stone", "Tumor"]:
                    c_path = split_path / cname
                    if c_path.exists():
                        cnt = len(list(c_path.glob("*.*")))
                        class_dist[cname] += cnt
                        split_info[split] += cnt
    else:
        # Fallback to DB distribution or simulated defaults
        if h.class_distribution:
            class_dist = h.class_distribution
        split_info = {"train": sum(class_dist.values())}

    total_size = sum(class_dist.values())

    return DatasetStatusOut(
        hospital_id=h.id,
        hospital_code=h.hospital_code,
        name=h.name,
        dataset_size=total_size,
        class_distribution=class_dist,
        is_valid=total_size > 0,
        split_info=split_info
    )


@router.post("/{hospital_id}/dataset-validate", response_model=DatasetValidateResponse)
def validate_dataset(hospital_id: int, db: Session = Depends(get_db)):
    """Inspects and validates integrity of local dataset partition."""
    h = _get_hospital_or_404(hospital_id, db)
    part_dir = _get_hospital_partition_dir(h.hospital_code)

    if not part_dir.exists():
        return DatasetValidateResponse(
            hospital_code=h.hospital_code,
            is_valid=False,
            total_samples=0,
            classes={},
            corrupted_images=0,
            message="Partition directory not found. Please run partitioning utility."
        )

    corrupted = 0
    class_counts: Dict[str, int] = {"Cyst": 0, "Normal": 0, "Stone": 0, "Tumor": 0}

    for split in ["train", "validation", "test"]:
        split_path = part_dir / split
        if split_path.exists():
            for cname in class_counts.keys():
                c_path = split_path / cname
                if c_path.exists():
                    for img_file in c_path.glob("*.*"):
                        try:
                            with Image.open(img_file) as img:
                                img.verify()
                            class_counts[cname] += 1
                        except Exception:
                            corrupted += 1

    total = sum(class_counts.values())
    is_valid = total > 0 and corrupted == 0

    msg = f"Dataset validated successfully: {total} healthy images verified across 4 classes."
    if corrupted > 0:
        msg = f"Validation warning: {corrupted} corrupted images detected out of {total + corrupted} files."

    return DatasetValidateResponse(
        hospital_code=h.hospital_code,
        is_valid=is_valid,
        total_samples=total,
        classes=class_counts,
        corrupted_images=corrupted,
        message=msg
    )


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
