"""
Hospital-scoped routes.
Drop at: backend/app/api/v1/routes/hospital.py
Mount in main.py:  app.include_router(hospital.router, prefix="/api/v1/hospital", tags=["hospital"])

No auth yet (per current project stage) — the frontend hospital selector
sends `hospital_id` explicitly on every request, and every query below
filters on it. This is the seam where real auth (JWT -> hospital_id) drops
in later without changing any query logic.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Hospital, Prediction
from app.schemas.dashboard import HospitalOut, PatientHistoryItem

router = APIRouter()


@router.get("/list", response_model=List[HospitalOut])
def list_hospitals(db: Session = Depends(get_db)):
    """Powers the hospital dropdown/selector in the UI."""
    return db.query(Hospital).filter(Hospital.is_active.is_(True)).order_by(Hospital.name).all()


def _get_hospital_or_404(hospital_id: int, db: Session) -> Hospital:
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Unknown hospital_id")
    return hospital


@router.get("/{hospital_id}/history", response_model=List[PatientHistoryItem])
def get_history(hospital_id: int, limit: int = 25, db: Session = Depends(get_db)):
    """Recent predictions for one hospital only — this is the isolation
    boundary: the query always filters on hospital_id, so one hospital's
    dashboard can never see another's records."""
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
