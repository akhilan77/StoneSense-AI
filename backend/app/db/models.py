"""
StoneSense-AI — Database Models
-------------------------------
Complete schema supporting:
1. Multi-hospital tenancy and isolation (hospitals, patients, predictions)
2. Federated Learning coordination & telemetry (federated_rounds, hospital_training_runs)
3. Model versioning registry & deployment pointers (model_versions)
4. Telemetry, system audit, and inference logs (inference_logs, system_logs, drift_records)

No raw patient CT images or sensitive identifiers are stored in the database.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Hospital(Base):
    """Registered hospital tenant in the federated network."""
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    hospital_code = Column(String(32), unique=True, index=True, nullable=False)  # e.g. "HOSP-001"
    name = Column(String(255), nullable=False)
    region = Column(String(128), nullable=True)
    is_active = Column(Boolean, default=True)
    dataset_size = Column(Integer, default=0)
    class_distribution = Column(JSON, nullable=True)  # {"Cyst": 100, "Normal": 200, ...}
    current_model_version = Column(String(64), nullable=True, default="resnet18_centralized_v1")
    created_at = Column(DateTime, default=datetime.utcnow)

    patients = relationship("Patient", back_populates="hospital")
    predictions = relationship("Prediction", back_populates="hospital")
    training_runs = relationship("HospitalTrainingRun", back_populates="hospital")
    update_logs = relationship("HospitalUpdateLog", back_populates="hospital")


class FederatedRound(Base):
    """Federated aggregation round records."""
    __tablename__ = "federated_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, unique=True, index=True, nullable=False)
    mode = Column(String(32), default="iid")  # "iid" | "non-iid"
    participants_count = Column(Integer, default=3)
    global_train_loss = Column(Float, nullable=True)
    global_train_acc = Column(Float, nullable=True)
    global_val_loss = Column(Float, nullable=True)
    global_val_acc = Column(Float, nullable=True)
    global_val_f1 = Column(Float, nullable=True)
    global_val_precision = Column(Float, nullable=True)
    global_val_recall = Column(Float, nullable=True)
    duration_sec = Column(Float, nullable=True)
    status = Column(String(32), default="completed")  # "completed" | "in_progress" | "failed"
    completed_at = Column(DateTime, default=datetime.utcnow)

    hospital_runs = relationship("HospitalTrainingRun", back_populates="federated_round", cascade="all, delete-orphan")
    model_version = relationship("ModelVersion", back_populates="federated_round", uselist=False)


class HospitalTrainingRun(Base):
    """Per-hospital local training telemetry submitted at each round."""
    __tablename__ = "hospital_training_runs"

    id = Column(Integer, primary_key=True, index=True)
    round_id = Column(Integer, ForeignKey("federated_rounds.id"), nullable=True, index=True)
    round_number = Column(Integer, index=True, nullable=False)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False, index=True)
    hospital_code = Column(String(32), nullable=False)
    train_loss = Column(Float, nullable=True)
    train_acc = Column(Float, nullable=True)
    train_f1 = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    val_acc = Column(Float, nullable=True)
    val_f1 = Column(Float, nullable=True)
    sample_count = Column(Integer, default=0)
    duration_sec = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital", back_populates="training_runs")
    federated_round = relationship("FederatedRound", back_populates="hospital_runs")


class ModelVersion(Base):
    """Registry of trained model checkpoints and deployment status."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_family = Column(String(32), nullable=False)   # "xgboost_risk" | "resnet18_ct"
    version_tag = Column(String(64), unique=True, index=True, nullable=False)     # "resnet18_fed_round_001"
    round_id = Column(Integer, ForeignKey("federated_rounds.id"), nullable=True)
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    mcc = Column(Float, nullable=True)
    is_deployed = Column(Boolean, default=False)
    artifact_path = Column(String(255), nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)

    federated_round = relationship("FederatedRound", back_populates="model_version")
    inference_logs = relationship("InferenceLog", back_populates="model_version")


class InferenceLog(Base):
    """High-frequency prediction latency and confidence logging."""
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True, index=True)
    prediction_type = Column(String(16), nullable=False)  # "risk" | "image"
    model_name = Column(String(64), nullable=False)
    version_tag = Column(String(64), nullable=True)
    result_label = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital")
    model_version = relationship("ModelVersion", back_populates="inference_logs")


class Patient(Base):
    """Clinical record strictly scoped to a single hospital."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False, index=True)
    reference_code = Column(String(64), nullable=False)  # de-identified reference code
    urine_features = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital", back_populates="patients")
    predictions = relationship("Prediction", back_populates="patient")


class Prediction(Base):
    """Clinical predictions for patient records."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    prediction_type = Column(String(16), nullable=False)  # "risk" | "image"
    model_name = Column(String(64), nullable=False)
    result_label = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    explainability_ref = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital", back_populates="predictions")
    patient = relationship("Patient", back_populates="predictions")


class HospitalUpdateLog(Base):
    """Federated model synchronization logs per hospital."""
    __tablename__ = "hospital_update_logs"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    status = Column(String(32), default="received")  # received | failed | pending
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital", back_populates="update_logs")
    model_version = relationship("ModelVersion")


class SystemLog(Base):
    """System operations and error monitoring logs."""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True, index=True)
    level = Column(String(16), nullable=False)   # info | warning | error
    message = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftRecord(Base):
    """Data & model drift snapshots."""
    __tablename__ = "drift_records"

    id = Column(Integer, primary_key=True, index=True)
    model_family = Column(String(32), nullable=False)
    drift_score = Column(Float, nullable=False)
    metric_name = Column(String(64), nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)
