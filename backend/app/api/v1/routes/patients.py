"""Patient records and patient-scoped assessment history."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Patient, Prediction
from app.schemas.patient import PatientCreate, PatientOut

router = APIRouter(prefix="/patients", tags=["patients"])


def _patient_out(patient: Patient, db: Session) -> PatientOut:
    profile = dict(patient.urine_features or {})
    identity = profile.get("identity", {})
    clinical = profile.get("clinical", {})
    latest: Dict[str, Any] = {}
    for prediction_type in ("risk", "image"):
        prediction = (
            db.query(Prediction)
            .filter(Prediction.patient_id == patient.id, Prediction.prediction_type == prediction_type)
            .order_by(Prediction.created_at.desc())
            .first()
        )
        if prediction:
            latest[prediction_type] = {
                "status": "completed",
                "label": prediction.result_label,
                "level": prediction.result_label,
                "score": round((prediction.confidence or 0) * 100),
                "result": prediction.result_label,
                "confidence": prediction.confidence,
                "created_at": prediction.created_at,
            }
    return PatientOut(
        id=patient.id,
        patient_id=patient.reference_code,
        name=patient.name or identity.get("name", patient.reference_code),
        phone=patient.phone or identity.get("phone", ""),
        blood_group=patient.blood_group or identity.get("blood_group", ""),
        admitted_date=patient.admitted_date or identity.get("admitted_date", ""),
        inspection_date=patient.last_inspected_date or identity.get("inspection_date"),
        clinical_profile=clinical,
        ml_risk=latest.get("risk", {"status": "pending"}),
        dl_imaging=latest.get("image", {"status": "pending"}),
    )


def _get_patient(patient_id: int, hospital_id: int, db: Session) -> Patient:
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.hospital_id == hospital_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found for this hospital.")
    return patient


@router.get("", response_model=List[PatientOut])
def list_patients(hospital_id: int = 1, db: Session = Depends(get_db)):
    patients = db.query(Patient).filter(Patient.hospital_id == hospital_id).order_by(Patient.id.desc()).all()
    return [_patient_out(patient, db) for patient in patients]


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: int, hospital_id: int = 1, db: Session = Depends(get_db)):
    return _patient_out(_get_patient(patient_id, hospital_id, db), db)


@router.post("", response_model=PatientOut, status_code=201)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    duplicate = db.query(Patient).filter(
        Patient.hospital_id == payload.hospital_id,
        Patient.reference_code == payload.reference_code,
    ).first()
    if duplicate:
        raise HTTPException(status_code=409, detail="Patient ID already exists for this hospital.")
    patient = Patient(
        hospital_id=payload.hospital_id,
        reference_code=payload.reference_code,
        name=payload.name,
        phone=payload.phone,
        blood_group=payload.blood_group,
        admitted_date=payload.admitted_date,
        last_inspected_date=payload.inspection_date,
        urine_features={"identity": {
            "name": payload.name,
            "phone": payload.phone,
            "blood_group": payload.blood_group,
            "admitted_date": payload.admitted_date,
            "inspection_date": payload.inspection_date,
        }},
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return _patient_out(patient, db)