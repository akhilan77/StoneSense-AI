"""
StoneSense-AI — Multi-hospital persistence models
---------------------------------------------------
Adds the data layer needed for:
  - per-hospital data isolation (every clinical record is tagged with hospital_id)
  - developer-side aggregate monitoring (model versions, drift, system logs,
    federated update history) WITHOUT ever exposing raw patient data to the
    developer role.

Drop this file at: backend/app/db/models.py
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Hospital(Base):
    """A registered hospital/tenant. Selected client-side via the hospital
    dropdown (no auth yet, per current project stage) but every downstream
    query is scoped by hospital_id so data never crosses tenants."""
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    hospital_code = Column(String(32), unique=True, index=True, nullable=False)  # e.g. "HOSP-001"
    name = Column(String(255), nullable=False)
    region = Column(String(128), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patients = relationship("Patient", back_populates="hospital")
    predictions = relationship("Prediction", back_populates="hospital")


class Patient(Base):
    """Clinical/urine input record. Scoped strictly to one hospital."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False, index=True)
    reference_code = Column(String(64), nullable=False)  # hospital's own de-identified patient ref
    urine_features = Column(JSON, nullable=False)  # specific_gravity, ph, osmolality, conductivity, urea, calcium
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital", back_populates="patients")
    predictions = relationship("Prediction", back_populates="patient")


class Prediction(Base):
    """Every ML risk prediction or DL image classification, tagged by
    hospital_id so developer aggregates can be computed without ever
    reading patient-identifying fields."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)

    prediction_type = Column(String(16), nullable=False)  # "risk" | "image"
    model_name = Column(String(64), nullable=False)        # "xgboost_risk_v2" | "resnet18_ct_v3"
    result_label = Column(String(64), nullable=True)        # risk_level or class_name
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    explainability_ref = Column(String(255), nullable=True)  # path/id to SHAP or Grad-CAM artifact
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("Hospital", back_populates="predictions")
    patient = relationship("Patient", back_populates="predictions")


class ModelVersion(Base):
    """Registry of trained model artifacts and their deployment status,
    shown on the Developer Dashboard."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_family = Column(String(32), nullable=False)   # "xgboost_risk" | "resnet18_ct"
    version_tag = Column(String(32), nullable=False)     # "v1.2.0"
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    mcc = Column(Float, nullable=True)
    is_deployed = Column(Boolean, default=False)
    artifact_path = Column(String(255), nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)


class HospitalUpdateLog(Base):
    """Federated update history — records each time a hospital pulled a
    new model version, so the developer dashboard can show 'received
    models / update history' per hospital."""
    __tablename__ = "hospital_update_logs"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    status = Column(String(32), default="received")  # received | failed | pending
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLog(Base):
    """Inference usage / error / uptime events, aggregated on the
    Developer Dashboard's 'System Monitoring' panel."""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True, index=True)
    level = Column(String(16), nullable=False)   # info | warning | error
    message = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftRecord(Base):
    """Periodic data/model drift snapshots per model family."""
    __tablename__ = "drift_records"

    id = Column(Integer, primary_key=True, index=True)
    model_family = Column(String(32), nullable=False)
    drift_score = Column(Float, nullable=False)   # 0.0 (no drift) - 1.0 (severe drift)
    metric_name = Column(String(64), nullable=False)  # e.g. "PSI", "KS-statistic"
    computed_at = Column(DateTime, default=datetime.utcnow)
