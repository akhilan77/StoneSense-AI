"""
One-off seed script — creates demo hospitals, a couple of model versions,
and sample system logs so both dashboards have something to show on first run.

Run with:  python -m app.db.seed   (from inside backend/, venv active)
Drop at: backend/app/db/seed.py
"""
from datetime import datetime, timedelta
import random

from app.db.database import SessionLocal, init_db
from app.db.models import Hospital, ModelVersion, HospitalUpdateLog, SystemLog, DriftRecord


def run():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Hospital).count() == 0:
            hospitals = [
                Hospital(hospital_code="HOSP-001", name="Apollo Renal Care", region="Chennai"),
                Hospital(hospital_code="HOSP-002", name="Fortis Nephrology Wing", region="Bengaluru"),
                Hospital(hospital_code="HOSP-003", name="AIIMS Urology Dept.", region="Delhi"),
            ]
            db.add_all(hospitals)
            db.commit()

        if db.query(ModelVersion).count() == 0:
            versions = [
                ModelVersion(model_family="xgboost_risk", version_tag="v1.0.0",
                             accuracy=0.8912, f1_score=0.8834, mcc=0.7801,
                             is_deployed=False, artifact_path="ml/models/xgb_risk_v1.pkl",
                             trained_at=datetime.utcnow() - timedelta(days=30)),
                ModelVersion(model_family="xgboost_risk", version_tag="v1.1.0",
                             accuracy=0.9167, f1_score=0.9091, mcc=0.8452,
                             is_deployed=True, artifact_path="ml/models/xgb_risk_v1_1.pkl",
                             trained_at=datetime.utcnow() - timedelta(days=5)),
                ModelVersion(model_family="resnet18_ct", version_tag="v2.0.0",
                             accuracy=0.9712, f1_score=0.9688, mcc=None,
                             is_deployed=False, artifact_path="dl/models/resnet18_v2.pth",
                             trained_at=datetime.utcnow() - timedelta(days=20)),
                ModelVersion(model_family="resnet18_ct", version_tag="v2.1.0",
                             accuracy=0.9850, f1_score=0.9793, mcc=None,
                             is_deployed=True, artifact_path="dl/models/resnet18_v2_1.pth",
                             trained_at=datetime.utcnow() - timedelta(days=2)),
            ]
            db.add_all(versions)
            db.commit()

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

        if db.query(SystemLog).count() == 0:
            for _ in range(15):
                db.add(SystemLog(
                    hospital_id=random.choice([None, 1, 2, 3]),
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
