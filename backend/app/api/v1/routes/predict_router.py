from fastapi import APIRouter, File, UploadFile

from app.schemas.patient import PatientInformation
from app.schemas.responses import (
    ExplainabilityResponse,
    RiskPredictionResponse,
    StoneDetectionResponse,
)
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/risk", response_model=RiskPredictionResponse)
def predict_risk(payload: PatientInformation) -> RiskPredictionResponse:
    """Return a mock risk prediction response for patient assessment.

    This endpoint is deliberately isolated from ML runtime behavior so the
    external contract can be validated before model training begins.
    """
    return PredictionService.get_mock_risk_prediction(payload)


@router.post("/image", response_model=StoneDetectionResponse)
def predict_image(image: UploadFile = File(...)) -> StoneDetectionResponse:
    """Return a mock image detection response.

    Args:
        image: Uploaded image file descriptor.

    Returns:
        StoneDetectionResponse: Mock stone detection payload.
    """
    _ = image.filename
    return PredictionService.get_mock_detection_response()


@router.post("/explain", response_model=ExplainabilityResponse)
def explain_prediction() -> ExplainabilityResponse:
    """Return a mock explainability payload for the current response contract."""
    return PredictionService.get_mock_explainability()
