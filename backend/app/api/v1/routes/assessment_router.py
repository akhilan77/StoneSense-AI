"""Assessment API endpoint router for combined multi-modal diagnostic assessments."""

import sys
import time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from app.schemas.patient import PatientInformation
from app.schemas.responses import FinalAssessmentResponse
from app.services.assessment_service import AssessmentService
from app.services.model_loader import model_loader

router = APIRouter(tags=["assessment"])


@router.post("/assessment", response_model=FinalAssessmentResponse)
async def create_assessment(
    patient_data: str = Form(..., description="JSON-serialized PatientInformation metadata"),
    image: Optional[UploadFile] = File(None)
) -> FinalAssessmentResponse:
    """Orchestrates combined assessment executing ML, DL, and CAM/SHAP explainability overlays."""
    # 1. Parse JSON metadata string into schema object
    import json
    try:
        patient_dict = json.loads(patient_data)
        patient = PatientInformation(**patient_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid patient_data JSON metadata payload: {str(e)}")

    image_bytes = None
    gradcam_path = None

    # 2. Check if image file exists
    if image is not None:
        try:
            image_bytes = await image.read()
            # Define standard location inside app's static files/outputs to write overlay
            output_dir = Path("backend/app/static/gradcam")
            output_dir.mkdir(parents=True, exist_ok=True)
            gradcam_path = str(output_dir / f"{image.filename}_overlay.png")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image upload bytes: {str(e)}")

    start_time = time.time()
    try:
        assessment = AssessmentService.build_assessment(
            patient_data=patient.dict(),
            image_bytes=image_bytes,
            gradcam_out_path=gradcam_path
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment engine execution failed: {str(e)}")

    elapsed = time.time() - start_time

    # Construct schema responses
    from app.schemas.responses import RiskPredictionResponse, StoneDetectionResponse, ShapExplanation
    
    risk_pred = RiskPredictionResponse(
        probability=assessment["risk_prediction"]["probability"],
        risk_level=assessment["risk_prediction"]["risk"],
        inference_time_sec=0.0
    )

    ct_pred = None
    if "class" in assessment["ct_prediction"]:
        ct_pred = StoneDetectionResponse(
            class_name=assessment["ct_prediction"]["class"],
            confidence=assessment["ct_prediction"]["confidence"]
        )

    shap_exp = ShapExplanation(
        top_features=assessment["shap"]["top_features"],
        feature_contributions=assessment["shap"]["feature_contributions"]
    )

    return FinalAssessmentResponse(
        patient=patient,
        risk_prediction=risk_pred,
        ct_prediction=ct_pred,
        gradcam=assessment["gradcam"],
        shap=shap_exp,
        recommendation=assessment["recommendation"],
        processing_time_sec=round(elapsed, 4)
    )
