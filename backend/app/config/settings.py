import os
from typing import Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application-level configuration for the backend service."""

    app_name: str = Field(default="StoneSense AI API")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(
        default=(
            "Explainable AI-Based Kidney Stone Detection & Risk Prediction "
            "System API contract"
        )
    )

    # Environment and Security Configuration
    environment: str = Field(
        default_factory=lambda: os.getenv("STONESENSE_ENV", os.getenv("ENVIRONMENT", "production")).strip().lower()
    )
    jwt_secret_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("STONESENSE_JWT_SECRET")
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)
    refresh_token_expire_days: int = Field(default=7)
    login_rate_limit_max: int = Field(default=10)
    login_rate_limit_window_sec: int = Field(default=60)

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
    gate_validation_data_source: str = Field(
        default="dl_partitions_iid_seed42_val",
        description="Default validation dataset signature for deployment comparisons.",
    )

    def get_jwt_secret(self) -> str:
        """Returns JWT secret key, enforcing explicit secret in production mode."""
        if self.jwt_secret_key:
            return self.jwt_secret_key
        if self.environment == "production":
            raise RuntimeError(
                "STONESENSE_JWT_SECRET environment variable is missing or empty. "
                "Production mode strictly forbids running with an unconfigured JWT secret."
            )
        return "dev-insecure-secret-key-stonesense-ai-2026-phase4"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()


