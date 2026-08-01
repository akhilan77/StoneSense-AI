from fastapi import APIRouter

from app.schemas.patient import PatientInformation
from app.schemas.responses import FinalAssessmentResponse
from app.services.assessment_service import AssessmentService

router = APIRouter(tags=["assessment"])


@router.post("/assess", response_model=FinalAssessmentResponse)
def assess(payload: PatientInformation) -> FinalAssessmentResponse:
    """Return a single merged JSON assessment contract.

    This endpoint composes patient information, risk prediction,
    image-detection, explainability, and recommendation into one
    top-level response object for the frontend integration contract.
    """
    return AssessmentService.build_assessment(payload)
