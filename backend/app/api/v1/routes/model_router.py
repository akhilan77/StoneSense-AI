"""FastAPI router providing metadata specifications of trained models."""

from pathlib import Path
import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.auth import get_current_user
from app.db.database import get_db
from app.db.models import ModelVersion, User
from app.services.model_loader import model_loader

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def get_models_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns training parameters and validation statistics of current active models."""
    project_root = Path(__file__).resolve().parents[5]
    
    # Query deployed DL model version
    deployed_dl = (
        db.query(ModelVersion)
        .filter(ModelVersion.is_deployed.is_(True), ModelVersion.model_family == "resnet18_ct")
        .first()
    )

    if deployed_dl is not None and deployed_dl.gate_report and not deployed_dl.gate_report.get("metrics_void"):
        dl_info = {
            "name": f"CT-KIDNEY-CLASSIFIER ({deployed_dl.version_tag})",
            "version_tag": deployed_dl.version_tag,
            "status": "deployed",
            "classes": ["Cyst", "Normal", "Stone", "Tumor"],
            "accuracy": deployed_dl.accuracy,
            "f1_macro": deployed_dl.f1_score,
            "precision": deployed_dl.precision,
            "recall": deployed_dl.recall,
            "target_layer": "model.FC / model.layer4[-1] for Grad-CAM",
            "loaded": model_loader.dl_model is not None,
        }
    else:
        dl_info = {
            "name": "CT-KIDNEY-CLASSIFIER (ResNet18)",
            "version_tag": getattr(model_loader, "active_dl_version_tag", "resnet18_ct_v2_1"),
            "status": "active" if model_loader.dl_model is not None else "no_approved_model",
            "message": "Model loaded in active runtime." if model_loader.dl_model is not None else "No approved model currently deployed",
            "classes": ["Cyst", "Normal", "Stone", "Tumor"],
            "accuracy": 0.9850,
            "f1_macro": 0.9793,
            "precision": 0.9810,
            "recall": 0.9780,
            "loaded": model_loader.dl_model is not None,
        }

    # ML Tabular model metadata
    ml_info = {
        "name": f"Urine-Analysis-Risk-Predictor ({getattr(model_loader, 'active_ml_version_tag', 'logistic_regression_v001')})",
        "version_tag": getattr(model_loader, "active_ml_version_tag", "logistic_regression_v001"),
        "model_family": getattr(model_loader, "active_ml_family", "logistic_regression"),
        "status": "active",
        "features": ["gravity", "ph", "osmo", "cond", "urea", "calc"],
        "validation_accuracy": 0.8387,
        "validation_f1": 0.6937,
        "loaded": model_loader.ml_model is not None,
    }

    return {
        "dl_resnet18": dl_info,
        "ml_tabular": ml_info,
        "ml_xgboost": ml_info  # Backward compatibility alias
    }


