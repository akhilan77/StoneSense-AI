from typing import Literal

from pydantic import BaseModel, Field


class PatientInformation(BaseModel):
    """Canonical request schema for risk-based kidney stone assessment.

    The structure intentionally mirrors the clinical conversational context
    that the API will accept before any ML or DL execution is performed.
    """

    age: int = Field(..., ge=0, le=120, description="Patient age in years.")
    gender: Literal["male", "female", "other"] = Field(
        ...,
        description="Biological sex or documented gender category.",
    )
    bmi: float = Field(..., ge=10.0, le=80.0, description="Body mass index.")
    blood_pressure: float = Field(..., ge=50.0, le=250.0, description="Systolic blood pressure reading.")
    diabetes: bool = Field(..., description="Whether the patient has diabetes.")
    family_history: bool = Field(..., description="Whether the patient has a family history of kidney stones.")
    water_intake: float = Field(..., ge=0.0, description="Daily water intake in liters.")
    urine_ph: float = Field(..., ge=0.0, le=14.0, description="Urine pH value.")
    urine_specific_gravity: float = Field(..., ge=1.0, le=1.1, description="Urine specific gravity.")
    calcium: float = Field(..., ge=0.0, description="Urinary calcium concentration.")
    uric_acid: float = Field(..., ge=0.0, description="Uric acid concentration.")
    creatinine: float = Field(..., ge=0.0, description="Serum creatinine concentration.")
