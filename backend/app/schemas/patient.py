from typing import Any, Dict, Literal, Optional

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
    osmolality: float = Field(default=0.0, ge=0.0, description="Urine osmolality used by the tabular model.")
    conductivity: float = Field(default=0.0, ge=0.0, description="Urine conductivity used by the tabular model.")
    urea: float = Field(default=0.0, ge=0.0, description="Urine urea used by the tabular model.")
    hospital_id: int = Field(default=1, description="Associated hospital ID for data scoping.")
    patient_id: Optional[int] = Field(default=None, description="Database patient ID to associate with the assessment.")


class PatientCreate(BaseModel):
    reference_code: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=255)
    phone: str = Field(..., min_length=1, max_length=64)
    blood_group: str = Field(..., min_length=1, max_length=8)
    admitted_date: str = Field(..., min_length=1, max_length=32)
    inspection_date: Optional[str] = Field(default=None, max_length=32)
    hospital_id: int = Field(default=1)


class PatientOut(BaseModel):
    id: int
    patient_id: str
    name: str
    phone: str
    blood_group: str
    admitted_date: str
    inspection_date: Optional[str] = None
    clinical_profile: Dict[str, Any] = Field(default_factory=dict)
    ml_risk: Optional[Dict[str, Any]] = None
    dl_imaging: Optional[Dict[str, Any]] = None

