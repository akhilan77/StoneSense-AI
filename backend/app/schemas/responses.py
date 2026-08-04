from typing import Literal, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.patient import PatientInformation


class RiskPredictionResponse(BaseModel):
    """Probability and risk classification for a patient-level assessment."""
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["Low", "High"]
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    inference_time_sec: float = Field(default=0.0)


class StoneDetectionResponse(BaseModel):
    """Stone-detection result payload for image-based diagnosis."""
    class_name: str = Field(..., description="CT Classification class label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_time_sec: float = Field(default=0.0)


class ShapExplanation(BaseModel):
    top_features: List[str] = Field(default_factory=list)
    feature_contributions: Dict[str, float] = Field(default_factory=dict)


class ExplainabilityResponse(BaseModel):
    """Explainability view for the final assessment contract."""
    shap_values: Dict[str, float] = Field(default_factory=dict)
    lime_explanation: str = Field(default="SHAP feature importance indicates primary biomarkers.")
    gradcam_image_url: str = Field(default="")


class FinalAssessmentResponse(BaseModel):
    """Top-level response that combines all domain-specific perspectives."""
    patient: Optional[PatientInformation] = None
    risk_prediction: RiskPredictionResponse
    ct_prediction: Optional[StoneDetectionResponse] = None
    gradcam: Optional[Dict[str, str]] = None
    shap: Optional[ShapExplanation] = None
    recommendation: str = Field(..., description="Human-readable clinical recommendation.")
    processing_time_sec: float = Field(default=0.0)
