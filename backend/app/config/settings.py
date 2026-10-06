from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application-level configuration for the backend service.

    These settings are intentionally lightweight because the current
    release focuses on architecture and mock API contract validation.
    """

    app_name: str = Field(default="StoneSense AI API")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(
        default=(
            "Explainable AI-Based Kidney Stone Detection & Risk Prediction "
            "System API contract"
        )
    )

    # Deployment Gate Configuration for DL Models
    gate_min_accuracy: float = Field(
        default=0.95,
        description="Minimum global accuracy required for model deployment eligibility.",
    )
    gate_min_f1: float = Field(
        default=0.94,
        description="Minimum macro F1-score required for model deployment eligibility.",
    )
    gate_max_regression_vs_current: float = Field(
        default=0.02,
        description="Maximum allowed performance regression against currently active deployed model on same validation split.",
    )
    gate_min_recall_stone: float = Field(
        default=0.96,
        description="Minimum per-class sensitivity (recall) for Stone class.",
    )
    gate_min_recall_tumor: float = Field(
        default=0.98,
        description="Minimum per-class sensitivity (recall) for Tumor class.",
    )


settings = Settings()

