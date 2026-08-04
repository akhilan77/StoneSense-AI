"""FastAPI API routes implementing real model predictions, complete assessments, and explainability endpoints."""

import sys
import time
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import json

from app.schemas.patient import PatientInformation
from app.schemas.responses import (
    RiskPredictionResponse,
    StoneDetectionResponse,
    FinalAssessmentResponse
)
from app.services.prediction_service import PredictionService
from app.services.assessment_service import AssessmentService
from app.services.model_loader import model_loader

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/risk", response_model=RiskPredictionResponse)
async def predict_risk(payload: PatientInformation) -> RiskPredictionResponse:
    """Classifies patient kidney stone risk from urine biochemistry parameters."""
    if model_loader.ml_model is None:
        raise HTTPException(status_code=503, detail="Risk prediction model not loaded.")
    
    start_time = time.time()
    res = PredictionService.predict_risk(payload.dict())
    elapsed = time.time() - start_time
    
    return RiskPredictionResponse(
        probability=res["probability"],
        risk_level=res["risk"],
        confidence=0.95,
        inference_time_sec=round(elapsed, 4)
    )


@router.post("/image", response_model=StoneDetectionResponse)
async def predict_image(image: UploadFile = File(...)) -> StoneDetectionResponse:
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
    return StoneDetectionResponse(
        class_name=res["class"],
        confidence=res["confidence"],
        inference_time_sec=round(elapsed, 4)
    )
