"""
Developer-scoped routes — aggregate, cross-hospital monitoring only.
Drop at: backend/app/api/v1/routes/developer.py
Mount in main.py:  app.include_router(developer.router, prefix="/api/v1/developer", tags=["developer"])

Privacy boundary: every query here reads model_versions / logs / drift
tables, or aggregates (COUNT/AVG) over predictions — it never selects
patient rows or urine_features. That's what keeps this dashboard
compliant with the "independently sourced, not patient-paired" limitation
noted in your README: the developer console reasons about model/system
health, never about an individual patient.
"""
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ModelVersion, HospitalUpdateLog, SystemLog, DriftRecord, Prediction, Hospital
from app.schemas.dashboard import (
    ModelPerformanceOut, HospitalUpdateLogOut, SystemLogOut,
    DriftPointOut, SystemMonitoringSummary, DeployModelRequest,
)

router = APIRouter()


@router.get("/model-performance", response_model=List[ModelPerformanceOut])
def model_performance(db: Session = Depends(get_db)):
    return db.query(ModelVersion).order_by(ModelVersion.trained_at.desc()).all()


@router.get("/model-versions", response_model=List[ModelPerformanceOut])
def model_versions(model_family: str | None = None, db: Session = Depends(get_db)):
    q = db.query(ModelVersion)
    if model_family:
        q = q.filter(ModelVersion.model_family == model_family)
    return q.order_by(ModelVersion.trained_at.desc()).all()


@router.post("/model-versions/deploy")
def deploy_model(payload: DeployModelRequest, db: Session = Depends(get_db)):
    """Marks one version as deployed and un-deploys siblings in the same
    model_family — this is the 'Download / Re-deploy models' action."""
    version = db.query(ModelVersion).filter(ModelVersion.id == payload.model_version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Unknown model_version_id")

    db.query(ModelVersion).filter(
        ModelVersion.model_family == version.model_family
    ).update({ModelVersion.is_deployed: False})
    version.is_deployed = True
    db.commit()
    return {"deployed": version.version_tag, "model_family": version.model_family}


@router.get("/hospital-logs", response_model=List[HospitalUpdateLogOut])
def hospital_logs(db: Session = Depends(get_db)):
    """Federated update history — 'received models / update history' panel."""
    rows = (
        db.query(HospitalUpdateLog, Hospital.name, ModelVersion.version_tag)
        .join(Hospital, Hospital.id == HospitalUpdateLog.hospital_id)
        .join(ModelVersion, ModelVersion.id == HospitalUpdateLog.model_version_id)
        .order_by(HospitalUpdateLog.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        HospitalUpdateLogOut(hospital_name=name, version_tag=tag, status=log.status, created_at=log.created_at)
        for log, name, tag in rows
    ]


@router.get("/system-monitoring", response_model=SystemMonitoringSummary)
def system_monitoring(db: Session = Depends(get_db)):
    since = datetime.utcnow() - timedelta(hours=24)
    total = db.query(func.count(Prediction.id)).filter(Prediction.created_at >= since).scalar() or 0
    errors = db.query(func.count(SystemLog.id)).filter(
        SystemLog.created_at >= since, SystemLog.level == "error"
    ).scalar() or 0
    avg_latency = db.query(func.avg(Prediction.latency_ms)).filter(Prediction.created_at >= since).scalar()

    # simple uptime proxy: 1 - (error events / total logged events) over 24h
    total_logs = db.query(func.count(SystemLog.id)).filter(SystemLog.created_at >= since).scalar() or 0
    uptime_pct = 100.0 if total_logs == 0 else round(100.0 * (1 - errors / max(total_logs, 1)), 2)

    return SystemMonitoringSummary(
        total_predictions_24h=total,
        error_count_24h=errors,
        avg_latency_ms=round(avg_latency, 2) if avg_latency else None,
        uptime_pct=uptime_pct,
    )


@router.get("/system-logs", response_model=List[SystemLogOut])
def system_logs(limit: int = 30, db: Session = Depends(get_db)):
    rows = (
        db.query(SystemLog, Hospital.name)
        .outerjoin(Hospital, Hospital.id == SystemLog.hospital_id)
        .order_by(SystemLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        SystemLogOut(hospital_name=name, level=log.level, message=log.message, created_at=log.created_at)
        for log, name in rows
    ]


@router.get("/drift-analysis", response_model=List[DriftPointOut])
def drift_analysis(model_family: str | None = None, db: Session = Depends(get_db)):
    q = db.query(DriftRecord)
    if model_family:
        q = q.filter(DriftRecord.model_family == model_family)
    rows = q.order_by(DriftRecord.computed_at.asc()).all()
    return [
        DriftPointOut(
            model_family=r.model_family, drift_score=r.drift_score,
            metric_name=r.metric_name, computed_at=r.computed_at,
        )
        for r in rows
    ]
