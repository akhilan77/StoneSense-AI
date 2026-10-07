"""Patient records and patient-scoped assessment history."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.auth import get_current_user, get_scoped_hospital_id, log_audit
from app.db.database import get_db
from app.db.models import Patient, Prediction, User
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
def list_patients(
    hospital_id: Optional[int] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lists patient records scoped strictly to the authenticated hospital."""
    if current_user.role == "developer":
        raise HTTPException(
            status_code=403,
            detail="Developers are not permitted to access patient PII endpoints.",
        )
    effective_hospital_id = get_scoped_hospital_id(current_user, hospital_id)
    patients = (
        db.query(Patient)
        .filter(Patient.hospital_id == effective_hospital_id)
        .order_by(Patient.id.desc())
        .all()
    )
    log_audit(
        db=db,
        action="PATIENT_READ",
        resource_type="patient_list",
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={"count": len(patients), "hospital_id": effective_hospital_id},
    )
    return [_patient_out(patient, db) for patient in patients]


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(
    patient_id: int,
    hospital_id: Optional[int] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetches a single patient record strictly within hospital authorization scope."""
    if current_user.role == "developer":
        raise HTTPException(
            status_code=403,
            detail="Developers are not permitted to access patient PII endpoints.",
        )
    effective_hospital_id = get_scoped_hospital_id(current_user, hospital_id)
    patient = _get_patient(patient_id, effective_hospital_id, db)
    log_audit(
        db=db,
        action="PATIENT_READ",
        resource_type="patient",
        resource_id=str(patient.id),
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={"patient_id": patient.id, "reference_code": patient.reference_code},
    )
    return _patient_out(patient, db)


@router.post("", response_model=PatientOut, status_code=201)
def create_patient(
    payload: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Creates a patient record in the authenticated hospital's partition."""
    if current_user.role == "developer":
        raise HTTPException(
            status_code=403,
            detail="Developers are not permitted to access patient PII endpoints.",
        )
    effective_hospital_id = get_scoped_hospital_id(current_user, payload.hospital_id)

    duplicate = db.query(Patient).filter(
        Patient.hospital_id == effective_hospital_id,
        Patient.reference_code == payload.reference_code,
    ).first()
    if duplicate:
        raise HTTPException(status_code=409, detail="Patient ID already exists for this hospital.")
    patient = Patient(
        hospital_id=effective_hospital_id,
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

    log_audit(
        db=db,
        action="PATIENT_CREATE",
        resource_type="patient",
        resource_id=str(patient.id),
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={"patient_id": patient.id, "reference_code": patient.reference_code},
    )

    return _patient_out(patient, db)