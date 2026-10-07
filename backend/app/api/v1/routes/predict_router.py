"""FastAPI API routes implementing real model predictions, complete assessments, and explainability endpoints."""

import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.auth import get_current_user, get_scoped_hospital_id, log_audit
from app.db.database import get_db
from app.db.models import InferenceLog, Patient, Prediction, User
from app.schemas.patient import PatientInformation
from app.schemas.responses import (
    RiskPredictionResponse,
    StoneDetectionResponse,
)
from app.services.explainability_service import (
    generate_gradcam_for_bytes,
    generate_shap_for_patient,
)
from app.services.model_loader import model_loader
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/risk", response_model=RiskPredictionResponse)
async def predict_risk(
    payload: PatientInformation,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RiskPredictionResponse:
    """Classifies patient kidney stone risk from urine biochemistry parameters."""
    if model_loader.ml_model is None:
        raise HTTPException(status_code=503, detail="Risk prediction model not loaded.")

    effective_hospital_id = get_scoped_hospital_id(current_user, payload.hospital_id)

    patient = None
    if payload.patient_id is not None:
        patient = db.query(Patient).filter(
            Patient.id == payload.patient_id,
            Patient.hospital_id == effective_hospital_id,
        ).first()
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found for this hospital.")

    start_time = time.time()
    res = PredictionService.predict_risk(payload.model_dump())
    elapsed = time.time() - start_time
    confidence = max(res["probability"], 1 - res["probability"])

    active_ml_ver = getattr(model_loader, "active_ml_version_tag", "logistic_regression_v001")
    active_ml_fam = getattr(model_loader, "active_ml_family", "logistic_regression")

    try:
        prediction = Prediction(
            hospital_id=effective_hospital_id,
            patient_id=patient.id if patient else None,
            prediction_type="risk",
            model_name=active_ml_ver,
            result_label=str(res["risk"]),
            confidence=confidence,
            latency_ms=elapsed * 1000,
        )
        inf_log = InferenceLog(
            hospital_id=effective_hospital_id,
            prediction_type="risk",
            model_name=active_ml_fam,
            version_tag=active_ml_ver,
            result_label=str(res["risk"]),
            confidence=confidence,
            latency_ms=elapsed * 1000,
        )
        db.add(prediction)
        db.add(inf_log)
        if patient:
            profile = dict(patient.urine_features or {})
            profile["clinical"] = payload.model_dump(exclude={"hospital_id", "patient_id"})
            patient.urine_features = profile
        db.commit()
    except Exception:
        db.rollback()

    log_audit(
        db=db,
        action="PREDICTION_RISK",
        resource_type="prediction",
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={
            "patient_id": patient.id if patient else None,
            "result_label": str(res["risk"]),
            "confidence": round(float(confidence), 4),
            "model_version": active_ml_ver,
        },
    )

    shap = generate_shap_for_patient(payload.model_dump())
    return RiskPredictionResponse(
        probability=res["probability"],
        risk_level=res["risk"],
        confidence=confidence,
        inference_time_sec=round(elapsed, 4),
        shap=shap,
    )


@router.post("/image", response_model=StoneDetectionResponse)
async def predict_image(
    image: UploadFile = File(...),
    hospital_id: Optional[int] = Form(None),
    patient_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StoneDetectionResponse:
    """Classifies CT scan slice using trained ResNet18 model singleton."""
    if model_loader.dl_model is None:
        raise HTTPException(status_code=503, detail="ResNet18 CT classification model not loaded.")

    effective_hospital_id = get_scoped_hospital_id(current_user, hospital_id)

    patient = None
    if patient_id is not None:
        patient = db.query(Patient).filter(
            Patient.id == patient_id,
            Patient.hospital_id == effective_hospital_id,
        ).first()
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found for this hospital.")

    start_time = time.time()
    try:
        content = await image.read()
        res = PredictionService.predict_ct_image(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image file: {str(e)}")

    elapsed = time.time() - start_time
    output_dir = Path(__file__).resolve().parents[3] / "static" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = None
    overlay_url = None
    gradcam_error = None
    try:
        overlay = generate_gradcam_for_bytes(content, str(output_dir / f"{uuid4().hex}_overlay.png"), target_class=res["class"])
        overlay_url = f"/static/gradcam/{Path(overlay['overlay_path']).name}"
    except Exception as exc:
        gradcam_error = str(exc)
        import logging
        logging.getLogger("predict_router").exception("Grad-CAM generation failed for CT upload")

    active_ver = getattr(model_loader, "active_dl_version_tag", "resnet18_ct_v2_1")

    try:
        prediction = Prediction(
            hospital_id=effective_hospital_id,
            patient_id=patient.id if patient else None,
            prediction_type="image",
            model_name=active_ver,
            result_label=str(res["class"]),
            confidence=res["confidence"],
            latency_ms=elapsed * 1000,
            explainability_ref=overlay_url,
        )
        inf_log = InferenceLog(
            hospital_id=effective_hospital_id,
            prediction_type="image",
            model_name="resnet18_ct",
            version_tag=active_ver,
            result_label=str(res["class"]),
            confidence=res["confidence"],
            latency_ms=elapsed * 1000,
        )
        db.add(prediction)
        db.add(inf_log)
        db.commit()
    except Exception:
        db.rollback()

    log_audit(
        db=db,
        action="PREDICTION_CT",
        resource_type="prediction",
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={
            "patient_id": patient.id if patient else None,
            "result_label": str(res["class"]),
            "confidence": round(float(res["confidence"]), 4),
            "model_version": active_ver,
        },
    )

    payload = {
        "class_name": res["class"],
        "confidence": res["confidence"],
        "inference_time_sec": round(elapsed, 4),
    }
    if overlay_url:
        payload["gradcam"] = {
            "overlay_url": overlay_url if res["class"] != "Normal" else "",
            "target_class": overlay.get("target_class", res["class"]),
            "available": bool(res["class"] != "Normal") and bool(overlay.get("available", True)),
            "message": (
                "No stone-specific localization is shown because the model classified this scan as Normal."
                if res["class"] == "Normal"
                else overlay.get("message", "")
            )
        }
    elif gradcam_error:
        payload["gradcam"] = {
            "overlay_url": "",
            "target_class": res["class"],
            "available": False,
            "message": "Explanation unavailable for this scan."
        }

    return StoneDetectionResponse(**payload)
