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
    prediction_id: Optional[int] = Field(default=None, description="Database ID of the persisted prediction record")


class GradCAMResponse(BaseModel):
    overlay_url: str = ""
    target_class: str = ""
    available: bool = False
    message: str = ""


class StoneDetectionResponse(BaseModel):
    """Independent CT image assessment output across multi-model DL architectures."""
    class_name: str = Field(..., description="CT Classification class label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_time_sec: float = Field(default=0.0)
    model_id: Optional[str] = Field(default="resnet18", description="Identifier of the model that produced prediction")
    model_name: Optional[str] = Field(default="ResNet18", description="Human-readable model name")
    gradcam: Optional[GradCAMResponse] = None
    status: str = Field(default="accepted", description="Review status: accepted, needs_review, or rejected_ood")
    review_reason: Optional[str] = Field(default=None, description="Detailed explanation if gate flagged image for review")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Raw softmax probability distribution across classes")
    ood_metrics: Optional[Dict[str, float]] = Field(default=None, description="OOD heuristic metrics including entropy and contrast")
    warnings: List[str] = Field(default_factory=list, description="Sanity warnings (e.g., non-axial aspect ratio or color photo)")
    prediction_id: Optional[int] = Field(default=None, description="Database ID of the persisted prediction record")

