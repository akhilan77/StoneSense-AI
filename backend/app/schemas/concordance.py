"""Pydantic schemas for multi-stream evidence concordance."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class ConcordanceOutcomeCode(str, Enum):
    """Closed set of stable machine codes for multi-stream concordance outcomes."""
    CONCORDANT_STONE_EVIDENCE = "CONCORDANT_STONE_EVIDENCE"
    CONCORDANT_NO_STONE_EVIDENCE = "CONCORDANT_NO_STONE_EVIDENCE"
    DISCORDANT = "DISCORDANT"
    NON_STONE_FINDING = "NON_STONE_FINDING"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    # Legacy alias for safe backward-compatible deserialization:
    CONCORDANT_LOW_CONCERN = "CONCORDANT_LOW_CONCERN"

    @classmethod
    def _missing_(cls, value: object) -> Optional["ConcordanceOutcomeCode"]:
        if isinstance(value, str) and value.upper() == "CONCORDANT_LOW_CONCERN":
            return cls.CONCORDANT_NO_STONE_EVIDENCE
        return None


class TabularRiskInput(BaseModel):
    """Tabular urine biochemistry prediction input for concordance evaluation."""
    probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Predicted risk probability [0.0, 1.0]")
    risk_level: Optional[str] = Field(default=None, description="Categorical risk label (e.g., Low, Moderate, High)")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Model prediction confidence score [0.0, 1.0]")


class ImagingOutputInput(BaseModel):
    """CT scan slice prediction output for concordance evaluation."""
    class_name: Optional[str] = Field(default=None, description="Predicted CT class: Cyst, Normal, Stone, Tumor")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top predicted class probability [0.0, 1.0]")
    probabilities: Optional[Dict[str, float]] = Field(default=None, description="Full softmax class distribution dictionary")
    status: Optional[str] = Field(default="accepted", description="Review gate status: accepted, needs_review, rejected_ood")


class ConcordanceRequest(BaseModel):
    """Request payload for evidence concordance evaluation.
    
    Accepts server-stored prediction references (risk_prediction_id, image_prediction_id)
    with tenant-level authorization. Deprecated direct payloads (risk, imaging) are only
    accepted if explicitly enabled in server settings.
    """
    risk_prediction_id: Optional[int] = Field(default=None, description="Database ID of server-stored tabular risk prediction")
    image_prediction_id: Optional[int] = Field(default=None, description="Database ID of server-stored CT imaging prediction")
    patient_id: Optional[int] = Field(default=None, description="Optional patient ID for strict tenant/patient authorization")

    # Deprecated direct payload fields (only permitted if allow_direct_concordance_payload=True)
    risk: Optional[TabularRiskInput] = Field(default=None, description="[DEPRECATED] Direct tabular urine risk assessment output")
    imaging: Optional[ImagingOutputInput] = Field(default=None, description="[DEPRECATED] Direct CT imaging classification output")


class ConcordanceResult(BaseModel):
    """Structured response from transparent rule-based concordance engine."""
    outcome_code: ConcordanceOutcomeCode = Field(..., description="Stable machine outcome code")
    outcome_label: str = Field(..., description="Short human-readable outcome label")
    explanation: str = Field(..., description="Plain-language clinical decision support explanation")
    rule_fired: str = Field(..., description="Identifier/name of the specific precedence rule that fired")
    clinician_review_recommended: bool = Field(..., description="Flag indicating whether clinician review is urgently or strongly advised")
    inputs_used: Dict[str, Any] = Field(default_factory=dict, description="Exact normalized inputs evaluated by the engine")
    thresholds_applied: Dict[str, Any] = Field(default_factory=dict, description="Configuration thresholds applied during evaluation")
    disclaimer: str = Field(..., description="Mandatory regulatory disclaimer concerning unpaired dataset training")

    @field_validator("outcome_code", mode="before")
    @classmethod
    def normalize_legacy_outcome_code(cls, v: Any) -> Any:
        if v == "CONCORDANT_LOW_CONCERN" or v == ConcordanceOutcomeCode.CONCORDANT_LOW_CONCERN:
            return ConcordanceOutcomeCode.CONCORDANT_NO_STONE_EVIDENCE
        return v
