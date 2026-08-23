from typing import Literal, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.patient import PatientInformation


class ShapExplanation(BaseModel):
    top_features: List[str] = Field(default_factory=list)
    feature_contributions: Dict[str, float] = Field(default_factory=dict)
    feature_directions: Dict[str, Literal["increases", "decreases", "neutral"]] = Field(default_factory=dict)
    summary: str = ""


class RiskPredictionResponse(BaseModel):
    """Independent clinical risk assessment output."""
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["Low", "High"]
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    inference_time_sec: float = Field(default=0.0)
    shap: Optional[ShapExplanation] = None


class StoneDetectionResponse(BaseModel):
    """Independent CT image assessment output."""
    class_name: str = Field(..., description="CT Classification class label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_time_sec: float = Field(default=0.0)
    gradcam: Optional[Dict[str, str]] = None


