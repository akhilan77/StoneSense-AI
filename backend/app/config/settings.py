import os
from typing import List, Optional
from pydantic import BaseModel, Field


def _parse_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:80",
        "http://127.0.0.1:80",
    ]


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

    # Database Configuration
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "STONESENSE_DATABASE_URL",
            os.getenv("DATABASE_URL", "sqlite:///stonesense.db")
        )
    )

    # Environment and Security Configuration
    environment: str = Field(
        default_factory=lambda: os.getenv(
            "STONESENSE_ENV",
            os.getenv("ENVIRONMENT", "production")
        ).strip().lower()
    )
    jwt_secret_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("STONESENSE_JWT_SECRET", os.getenv("JWT_SECRET_KEY"))
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)
    refresh_token_expire_days: int = Field(default=7)
    login_rate_limit_max: int = Field(default=10)
    login_rate_limit_window_sec: int = Field(default=60)

    # CORS Configuration
    cors_origins: List[str] = Field(default_factory=_parse_cors_origins)

    # Deployment Gate Configuration for DL Models
    gate_min_accuracy: float = Field(
        default_factory=lambda: float(os.getenv("GATE_MIN_ACCURACY", "0.95")),
        description="Minimum global accuracy required for model deployment eligibility.",
    )
    gate_min_f1: float = Field(
        default_factory=lambda: float(os.getenv("GATE_MIN_F1", "0.94")),
        description="Minimum macro F1-score required for model deployment eligibility.",
    )
    gate_max_regression_vs_current: float = Field(
        default_factory=lambda: float(os.getenv("GATE_MAX_REGRESSION_VS_CURRENT", "0.02")),
        description="Maximum allowed performance regression against currently active deployed model on same validation split.",
    )
    gate_min_recall_stone: float = Field(
        default_factory=lambda: float(os.getenv("GATE_MIN_RECALL_STONE", "0.96")),
        description="Minimum per-class sensitivity (recall) for Stone class.",
    )
    gate_min_recall_tumor: float = Field(
        default_factory=lambda: float(os.getenv("GATE_MIN_RECALL_TUMOR", "0.98")),
        description="Minimum per-class sensitivity (recall) for Tumor class.",
    )
    gate_validation_data_source: str = Field(
        default_factory=lambda: os.getenv("GATE_VALIDATION_DATA_SOURCE", "dl_partitions_iid_seed42_val"),
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
