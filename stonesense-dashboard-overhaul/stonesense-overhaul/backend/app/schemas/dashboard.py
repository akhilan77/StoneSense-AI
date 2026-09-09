"""
Pydantic v2 schemas for the two new dashboards.
Drop at: backend/app/schemas/dashboard.py
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


# ---------- Hospital-scoped ----------

class HospitalOut(BaseModel):
    id: int
    hospital_code: str
    name: str
    region: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True


class PatientHistoryItem(BaseModel):
    id: int
    reference_code: str
    prediction_type: str
    model_name: str
    result_label: Optional[str]
    confidence: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- Developer-scoped ----------

class ModelPerformanceOut(BaseModel):
    model_family: str
    version_tag: str
    accuracy: Optional[float]
    f1_score: Optional[float]
    mcc: Optional[float]
    is_deployed: bool
    trained_at: datetime

    class Config:
        from_attributes = True


class HospitalUpdateLogOut(BaseModel):
    hospital_name: str
    version_tag: str
    status: str
    created_at: datetime


class SystemLogOut(BaseModel):
    hospital_name: Optional[str]
    level: str
    message: str
    created_at: datetime


class DriftPointOut(BaseModel):
    model_family: str
    drift_score: float
    metric_name: str
    computed_at: datetime


class SystemMonitoringSummary(BaseModel):
    total_predictions_24h: int
    error_count_24h: int
    avg_latency_ms: Optional[float]
    uptime_pct: float


class DeployModelRequest(BaseModel):
    model_version_id: int
