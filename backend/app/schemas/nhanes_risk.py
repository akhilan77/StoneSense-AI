"""Schemas for the NHANES clinical-profile model comparison endpoint (POST /api/v1/predict/risk/compare)."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NhanesCompareRequest(BaseModel):
    """Form values keyed by NHANES variable name (see GET /predict/risk/compare/spec).

    The allowed keys, ranges and answer options come from ml/artifacts/nhanes_v003/feature_spec.json and are
    validated server side. Optional items may be omitted or null (they are imputed with training medians).
    The values are used for scoring only and are not stored.
    """

    features: Dict[str, Optional[float]] = Field(..., description="Feature values keyed by NHANES variable name.")
    hospital_id: int = Field(default=1, description="Associated hospital ID for data scoping.")
    patient_id: Optional[int] = Field(default=None, description="Database patient ID this assessment is run for.")
    operating_point: Literal["accuracy", "balanced"] = Field(
        default="accuracy",
        description="Operating point the headline risk level is reported at: 'accuracy' (max sensitivity s.t. validation accuracy >= 72%) "
                    "or 'balanced' (max balanced accuracy). Both are always returned per model.",
    )


class TopFactor(BaseModel):
    feature: str
    label: str
    value: Optional[float] = None
    value_label: Optional[str] = None
    contribution: float
    direction: Literal["increases", "decreases", "neutral"]


class ModelTestMetrics(BaseModel):
    """Stored held-out test metrics of a registered model (NOT this patient's accuracy)."""

    evaluated_on: str
    test_cycle: str
    n: int
    prevalence: float
    majority_baseline_accuracy: float
    roc_auc: float
    roc_auc_ci95: List[float]
    pr_auc: Optional[float] = None
    brier: Optional[float] = None
    at_accuracy_threshold: Optional[Dict[str, Any]] = None
    at_balanced_threshold: Optional[Dict[str, Any]] = None
    # legacy nhanes_v003 entries
    at_balanced_accuracy_threshold: Optional[Dict[str, Any]] = None
    at_accuracy_ge_75_threshold: Optional[Dict[str, Any]] = None


class OperatingPointResult(BaseModel):
    """One operating point of one model: the validation-chosen threshold, this patient's risk level at it and the
    stored held-out TEST metrics at it (accuracy, sensitivity, specificity, balanced accuracy, ... with 95% CIs)."""

    label: str
    rule: str
    threshold: Optional[float] = None
    risk_level: Optional[Literal["Low", "High"]] = None
    validation: Optional[Dict[str, Any]] = None
    test: Optional[Dict[str, Any]] = None


class ModelComparisonResult(BaseModel):
    model_id: str
    model_family: str
    model_name: str
    probability: float
    risk_level: Literal["Low", "High"]   # at the selected operating point
    threshold: float                     # threshold of the selected operating point
    operating_point: Literal["accuracy", "balanced"] = "accuracy"
    operating_points: Dict[str, OperatingPointResult] = Field(default_factory=dict)
    imbalance_strategy: Optional[str] = None
    top_factors: List[TopFactor]
    explanation_summary: str
    test_metrics: ModelTestMetrics
    status: str
    variant: str


class NhanesCompareResponse(BaseModel):
    models: List[ModelComparisonResult]
    average_probability: float
    models_agree: bool
    n_high: int
    n_low: int
    operating_point: Literal["accuracy", "balanced"] = "accuracy"
    n_high_by_point: Dict[str, int] = Field(default_factory=dict)
    probability_min: float
    probability_max: float
    warnings: List[str] = Field(default_factory=list)
    metrics_note: str
    disclaimer: str
    feature_set: str
    inference_time_sec: float
