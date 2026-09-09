"""Pydantic v2 schemas for Federated Learning endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class FederatedOverviewOut(BaseModel):
    current_round: int
    global_accuracy: float
    global_f1: float
    global_loss: float
    global_precision: float
    global_recall: float
    active_hospitals_count: int
    current_model_version: str
    total_samples: int
    last_updated: Optional[datetime] = None


class HospitalRunTelemetryOut(BaseModel):
    id: int
    round_number: int
    hospital_id: int
    hospital_code: str
    hospital_name: Optional[str] = None
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    train_f1: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    val_f1: Optional[float] = None
    sample_count: int
    duration_sec: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class FederatedRoundDetailOut(BaseModel):
    id: int
    round_number: int
    mode: str
    participants_count: int
    global_train_loss: Optional[float] = None
    global_train_acc: Optional[float] = None
    global_val_loss: Optional[float] = None
    global_val_acc: Optional[float] = None
    global_val_f1: Optional[float] = None
    global_val_precision: Optional[float] = None
    global_val_recall: Optional[float] = None
    duration_sec: Optional[float] = None
    status: str
    completed_at: datetime
    hospital_runs: List[HospitalRunTelemetryOut] = []

    class Config:
        from_attributes = True


class HospitalParticipationOut(BaseModel):
    hospital_id: int
    hospital_code: str
    name: str
    region: Optional[str] = None
    dataset_size: int
    total_rounds_participated: int
    sample_contribution_pct: float
    latest_local_f1: Optional[float] = None
    current_model_version: Optional[str] = None


class DatasetStatusOut(BaseModel):
    hospital_id: int
    hospital_code: str
    name: str
    dataset_size: int
    class_distribution: Dict[str, int]
    is_valid: bool = True
    split_info: Dict[str, int] = {}


class DatasetValidateResponse(BaseModel):
    hospital_code: str
    is_valid: bool
    total_samples: int
    classes: Dict[str, int]
    corrupted_images: int = 0
    message: str


class LocalTrainingTriggerResponse(BaseModel):
    hospital_code: str
    status: str
    message: str
    metrics: Optional[Dict[str, float]] = None


class FederatedStatusOut(BaseModel):
    hospital_code: str
    status: str
    current_round: int
    current_model_version: str
    last_round_participated: Optional[int] = None
    local_accuracy: Optional[float] = None
    local_f1: Optional[float] = None
