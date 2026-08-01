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


settings = Settings()
