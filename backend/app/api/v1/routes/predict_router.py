"""FastAPI API routes implementing real model predictions, complete assessments, and explainability endpoints."""

import time
from pathlib import Path
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import uuid4

from app.db.database import get_db
from app.db.models import Prediction
from app.schemas.patient import PatientInformation
from app.schemas.responses import (
    RiskPredictionResponse,
    StoneDetectionResponse,
)
from app.services.prediction_service import PredictionService
from app.services.model_loader import model_loader
from app.services.explainability_service import generate_shap_for_patient, generate_gradcam_for_bytes

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/risk", response_model=RiskPredictionResponse)
async def predict_risk(
    payload: PatientInformation,
    db: Session = Depends(get_db),
) -> RiskPredictionResponse:
    """Classifies patient kidney stone risk from urine biochemistry parameters."""
    if model_loader.ml_model is None:
        raise HTTPException(status_code=503, detail="Risk prediction model not loaded.")

    start_time = time.time()
    res = PredictionService.predict_risk(payload.model_dump())
    elapsed = time.time() - start_time
    confidence = max(res["probability"], 1 - res["probability"])

    try:
        prediction = Prediction(
            hospital_id=payload.hospital_id,
            prediction_type="risk",
            model_name="xgboost_risk_v1_1",
            result_label=str(res["risk"]),
            confidence=confidence,
            latency_ms=elapsed * 1000,
        )
        db.add(prediction)
        db.commit()
    except Exception:
        db.rollback()

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
    hospital_id: int = Form(1),
    db: Session = Depends(get_db),
) -> StoneDetectionResponse:
    """Classifies CT scan slice using trained ResNet18 model singleton."""
    if model_loader.dl_model is None:
        raise HTTPException(status_code=503, detail="ResNet18 CT classification model not loaded.")

    start_time = time.time()
    try:
        content = await image.read()
        res = PredictionService.predict_ct_image(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image file: {str(e)}")

    elapsed = time.time() - start_time
    output_dir = Path(__file__).resolve().parents[4] / "static" / "gradcam"
    overlay = generate_gradcam_for_bytes(content, str(output_dir / f"{uuid4().hex}_overlay.png"))

    active_ver = getattr(model_loader, "active_dl_version_tag", "resnet18_ct_v2_1")

    try:
        from app.db.models import InferenceLog
        prediction = Prediction(
            hospital_id=hospital_id,
            prediction_type="image",
            model_name=active_ver,
            result_label=str(res["class"]),
            confidence=res["confidence"],
            latency_ms=elapsed * 1000,
            explainability_ref=f"/static/gradcam/{Path(overlay['overlay_path']).name}",
        )
        inf_log = InferenceLog(
            hospital_id=hospital_id,
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

    return StoneDetectionResponse(
        class_name=res["class"],
        confidence=res["confidence"],
        inference_time_sec=round(elapsed, 4),
        gradcam={"overlay_url": f"/static/gradcam/{Path(overlay['overlay_path']).name}"}
    )

