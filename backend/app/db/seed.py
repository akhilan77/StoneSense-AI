"""
One-off seed script — creates demo hospitals, model versions, sample predictions,
and system logs so both dashboards have something to show on first run.

Run with:  python -m app.db.seed   (from inside backend/, venv active)
Drop at: backend/app/db/seed.py
"""
from datetime import datetime, timedelta
import random

from app.config.settings import settings
from app.core.security import hash_password
from app.db.database import SessionLocal, init_db
from app.db.models import (
    DriftRecord,
    Hospital,
    HospitalUpdateLog,
    ModelVersion,
    Patient,
    Prediction,
    SystemLog,
    User,
)


def run():
    init_db()
    db = SessionLocal()
    try:
        # 1. Seed Hospitals first to satisfy Foreign Key constraints on users/patients
        demo_hospitals = [
            {"hospital_code": "HOSP-001", "name": "Apollo Renal Care", "region": "Chennai"},
            {"hospital_code": "HOSP-002", "name": "Fortis Nephrology Wing", "region": "Bengaluru"},
            {"hospital_code": "HOSP-003", "name": "AIIMS Urology Dept.", "region": "Delhi"},
        ]
        for h_data in demo_hospitals:
            existing_h = db.query(Hospital).filter_by(hospital_code=h_data["hospital_code"]).first()
            if not existing_h:
                db.add(Hospital(**h_data))
        db.commit()

        # Build lookup mapping for hospital codes to primary key IDs
        hosp_map = {h.hospital_code: h.id for h in db.query(Hospital).all()}
        hosp1_id = hosp_map.get("HOSP-001")
        hosp2_id = hosp_map.get("HOSP-002")
        hosp3_id = hosp_map.get("HOSP-003")

        # 2. Seed demo users only in non-production environments
        if not settings.is_production:
            demo_users = [
                {
                    "email": "admin@stonesense.ai",
                    "password": "StoneSenseAdmin!2026",
                    "role": "admin",
                    "hospital_id": None,
                },
                {
                    "email": "dev@stonesense.ai",
                    "password": "StoneSenseDev!2026",
                    "role": "developer",
                    "hospital_id": None,
                },
                {
                    "email": "hospital1@stonesense.ai",
                    "password": "Hospital1!2026",
                    "role": "hospital_user",
                    "hospital_id": hosp1_id,
                },
                {
                    "email": "hospital2@stonesense.ai",
                    "password": "Hospital2!2026",
                    "role": "hospital_user",
                    "hospital_id": hosp2_id,
                },
                {
                    "email": "hospital3@stonesense.ai",
                    "password": "Hospital3!2026",
                    "role": "hospital_user",
                    "hospital_id": hosp3_id,
                },
            ]
            for u in demo_users:
                existing = db.query(User).filter(User.email == u["email"]).first()
                if not existing:
                    user = User(
                        email=u["email"],
                        password_hash=hash_password(u["password"]),
                        role=u["role"],
                        hospital_id=u["hospital_id"],
                        is_active=True,
                    )
                    db.add(user)
            db.commit()

        # 3. Seed Patients
        if hosp1_id and db.query(Patient).count() == 0:
            patients = [
                Patient(hospital_id=hosp1_id, reference_code="PT-2201", urine_features={}),
                Patient(hospital_id=hosp1_id, reference_code="PT-2198", urine_features={}),
                Patient(hospital_id=hosp1_id, reference_code="PT-2190", urine_features={}),
                Patient(hospital_id=hosp1_id, reference_code="PT-2183", urine_features={}),
            ]
            db.add_all(patients)
            db.commit()

        # 4. Seed Predictions
        if hosp1_id and db.query(Prediction).count() == 0:
            patients = db.query(Patient).order_by(Patient.id.asc()).all()
            sample_predictions = [
                Prediction(hospital_id=hosp1_id, patient_id=patients[0].id if len(patients) > 0 else None, prediction_type="image", model_name="resnet18_ct", result_label="Stone", confidence=0.942, latency_ms=64.0, created_at=datetime.utcnow() - timedelta(minutes=10)),
                Prediction(hospital_id=hosp1_id, patient_id=patients[1].id if len(patients) > 1 else None, prediction_type="risk", model_name="xgboost_risk", result_label="Low", confidence=0.881, latency_ms=45.0, created_at=datetime.utcnow() - timedelta(minutes=45)),
                Prediction(hospital_id=hosp1_id, patient_id=patients[2].id if len(patients) > 2 else None, prediction_type="image", model_name="resnet18_ct", result_label="Normal", confidence=0.974, latency_ms=58.0, created_at=datetime.utcnow() - timedelta(hours=2)),
                Prediction(hospital_id=hosp1_id, patient_id=patients[3].id if len(patients) > 3 else None, prediction_type="risk", model_name="xgboost_risk", result_label="High", confidence=0.910, latency_ms=52.0, created_at=datetime.utcnow() - timedelta(hours=4)),
            ]
            db.add_all(sample_predictions)
            db.commit()

        # 5. Seed Model Versions
        if db.query(ModelVersion).count() == 0:
            versions = [
                ModelVersion(model_family="xgboost_risk", version_tag="v1.0.0",
                             accuracy=0.8912, f1_score=0.8834, mcc=0.7801,
                             is_deployed=False, status="archived", gate_report={"legacy": True},
                             artifact_path="ml/models/xgb_risk_v1.pkl",
                             trained_at=datetime.utcnow() - timedelta(days=30)),
                ModelVersion(model_family="xgboost_risk", version_tag="v1.1.0",
                             accuracy=0.9167, f1_score=0.9091, mcc=0.8452,
                             is_deployed=True, status="deployed", deployed_at=datetime.utcnow() - timedelta(days=5),
                             approved_by="system_seed", approved_at=datetime.utcnow() - timedelta(days=5),
                             artifact_path="ml/models/xgb_risk_v1_1.pkl",
                             trained_at=datetime.utcnow() - timedelta(days=5)),
                ModelVersion(model_family="resnet18_ct", version_tag="v2.0.0",
                             accuracy=0.9712, f1_score=0.9688, mcc=None,
                             is_deployed=False, status="archived",
                             gate_report={"legacy": True, "metrics_void": True, "trained_on_leaky_partitions": True},
                             artifact_path="dl/models/resnet18_v2.pth",
                             trained_at=datetime.utcnow() - timedelta(days=20)),
                ModelVersion(model_family="resnet18_ct", version_tag="v2.1.0",
                             accuracy=0.9850, f1_score=0.9793, mcc=None,
                             is_deployed=False, status="archived",
                             gate_report={"legacy": True, "metrics_void": True, "trained_on_leaky_partitions": True, "note": "Unverified legacy slice-level split"},
                             artifact_path="dl/models/resnet18_v2_1.pth",
                             trained_at=datetime.utcnow() - timedelta(days=2)),
                ModelVersion(model_family="resnet18_ct", version_tag="resnet18_grouped_audit_v1",
                             accuracy=0.8694, f1_score=0.8251, precision=0.8319, recall=0.8378, mcc=None,
                             is_deployed=False, status="pending_review",
                             gate_report={
                                 "audited": True,
                                 "data_source": "grouped_split_pHash_d2",
                                 "slice_accuracy": 0.8694,
                                 "slice_f1": 0.8251,
                                 "cluster_accuracy": 0.9648,
                                 "cluster_f1": 0.9547,
                                 "stone_recall": 0.6570,
                                 "tumor_recall": 1.0,
                                 "patient_level_verified": False,
                                 "disclaimer": "Research Prototype Only — Not for Clinical Diagnosis.",
                             },
                             artifact_path="dl/models/kidney_resnet18.pth",
                             trained_at=datetime.utcnow() - timedelta(hours=1)),
            ]
            db.add_all(versions)
            db.commit()

        # 6. Seed Hospital Update Logs
        if db.query(HospitalUpdateLog).count() == 0:
            hospitals = db.query(Hospital).all()
            deployed = db.query(ModelVersion).filter_by(is_deployed=True).all()
            for h in hospitals:
                for v in deployed:
                    db.add(HospitalUpdateLog(
                        hospital_id=h.id, model_version_id=v.id,
                        status=random.choice(["received", "received", "pending"]),
                        created_at=datetime.utcnow() - timedelta(days=random.randint(0, 4)),
                    ))
            db.commit()

        # 7. Seed System Logs
        if db.query(SystemLog).count() == 0:
            valid_hospital_ids = [None] + [h.id for h in db.query(Hospital).all()]
            for _ in range(15):
                db.add(SystemLog(
                    hospital_id=random.choice(valid_hospital_ids),
                    level=random.choice(["info", "info", "warning", "error"]),
                    message=random.choice([
                        "Inference request completed",
                        "CT image below expected resolution — upscaled",
                        "Model artifact reload triggered",
                        "Prediction latency exceeded 150ms threshold",
                    ]),
                    created_at=datetime.utcnow() - timedelta(hours=random.randint(0, 72)),
                ))
            db.commit()

        # 8. Seed Drift Records
        if db.query(DriftRecord).count() == 0:
            for family in ["xgboost_risk", "resnet18_ct"]:
                for i in range(6):
                    db.add(DriftRecord(
                        model_family=family,
                        drift_score=round(random.uniform(0.02, 0.18), 3),
                        metric_name="PSI" if family == "xgboost_risk" else "KS-statistic",
                        computed_at=datetime.utcnow() - timedelta(days=i * 5),
                    ))
            db.commit()

        print("Seed complete.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
