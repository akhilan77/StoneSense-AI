from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.patient import PatientInformation


class RiskPredictionResponse(BaseModel):
    """Probability and risk classification for a patient-level assessment."""

    probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class StoneDetectionResponse(BaseModel):
    """Stone-detection result payload for image-based diagnosis."""

    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    stone_size: str = Field(..., description="Human-readable stone size category.")
    stone_location: str = Field(..., description="Detected stone anatomical location.")


class ExplainabilityResponse(BaseModel):
    """Explainability view for the final assessment contract."""

    shap_values: dict[str, float] = Field(default_factory=dict)
    lime_explanation: str = Field(default="LIME explanation pending model integration.")
    gradcam_image_url: str = Field(default="https://example.com/mock-gradcam.png")


class FinalAssessmentResponse(BaseModel):
    """Top-level response that combines all domain-specific perspectives."""

    patient: PatientInformation
    risk_prediction: RiskPredictionResponse
    stone_detection: StoneDetectionResponse
    explainability: ExplainabilityResponse
    recommendation: str = Field(..., description="Human-readable clinical recommendation.")
