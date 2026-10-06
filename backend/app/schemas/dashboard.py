"""
Pydantic v2 schemas for the two new dashboards.
Drop at: backend/app/schemas/dashboard.py
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
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
    id: int
    model_family: str
    version_tag: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    mcc: Optional[float] = None
    is_deployed: bool
    status: str = "pending_review"
    gate_report: Optional[Dict[str, Any]] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    previous_deployed_version_id: Optional[int] = None
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
    approved_by: Optional[str] = "developer_admin"


class RollbackModelRequest(BaseModel):
    model_family: str = "resnet18_ct"
    target_version_id: Optional[int] = None
    approved_by: Optional[str] = "rollback_admin"


class MLTrainingResponse(BaseModel):
    status: str
    model_family: str
    model_name: str
    version_tag: str
    artifact_path: str
    metrics: Dict[str, float]
    duration_sec: float
    message: str
