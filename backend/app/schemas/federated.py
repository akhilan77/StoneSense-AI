"""Pydantic v2 schemas for DL Federated Learning endpoints."""

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
    model_version: Optional[str] = None
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
    dataset_version: Optional[str] = None
    last_updated: Optional[datetime] = None
    last_validated: Optional[datetime] = None
    validation_message: Optional[str] = None


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


class StartRoundRequest(BaseModel):
    num_rounds: int = 1
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.0005
    mode: str = "iid"


class StartRoundResponse(BaseModel):
    round: int
    status: str
    global_model_version: str
    message: Optional[str] = None


class ClientLiveStatus(BaseModel):
    hospital_id: str
    hospital_name: Optional[str] = None
    status: str
    samples: Optional[int] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    loss: Optional[float] = None
    duration_sec: Optional[float] = None


class RoundLiveStatusResponse(BaseModel):
    round: int
    status: str  # "READY" | "ROUND_STARTED" | "GLOBAL_MODEL_DISTRIBUTING" | "LOCAL_TRAINING" | "FEDAVG_STARTED" | "COMPLETED" | "FAILED"
    previous_model_version: Optional[str] = None
    global_model_version: str
    participating_hospitals: int = 3
    completed_hospitals: int = 0
    clients: List[ClientLiveStatus] = []
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class HospitalLiveStatusResponse(BaseModel):
    hospital_id: str
    round: int
    status: str
    phase: Optional[str] = None
    global_model_version: str
    local_training: Optional[Dict[str, Any]] = None
    update_submitted: bool = False
    model_updated: bool = False
    last_event: Optional[str] = None
    last_event_at: Optional[datetime] = None
    round_status: Optional[str] = None


class FederatedEventMessage(BaseModel):
    event: str
    round: int
    hospital_id: Optional[str] = None
    model_version: Optional[str] = None
    status: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

