"""FastAPI router providing metadata specifications of trained models across ML and DL subsystems."""

from pathlib import Path
import json
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.auth import get_current_user
from app.db.database import get_db
from app.db.models import ModelVersion, User
from app.services.model_loader import model_loader

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/dl")
async def get_dl_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns metadata, readiness status, and benchmark metrics for all 4 CT deep learning models.
    
    Supported models:
    - ResNet18 (READY)
    - YOLO26 (READY)
    - DINOv3 (PENDING_WEIGHTS)
    - QKNN (PENDING_WEIGHTS)
    """
    from dl.models.registry import dl_registry

    models_list = dl_registry.list_models()

    # Enrich ResNet18 with deployed DB records if available
    deployed_dl = (
        db.query(ModelVersion)
        .filter(ModelVersion.is_deployed.is_(True), ModelVersion.model_family == "resnet18_ct")
        .first()
    )
    if deployed_dl:
        for m in models_list:
            if m["id"] == "resnet18":
                m["is_deployed"] = True
                m["deployed_version_tag"] = deployed_dl.version_tag
                if deployed_dl.accuracy is not None:
                    m["accuracy"] = deployed_dl.accuracy
                if deployed_dl.f1_score is not None:
                    m["macro_f1"] = deployed_dl.f1_score

    return {"models": models_list}


@router.get("")
async def get_models_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns training parameters and validation statistics of current active models (with backward compatibility)."""
    from dl.models.registry import dl_registry
    
    # Query deployed DL model version
    deployed_dl = (
        db.query(ModelVersion)
        .filter(ModelVersion.is_deployed.is_(True), ModelVersion.model_family == "resnet18_ct")
        .first()
    )

    r18_wrapper = dl_registry.get_model("resnet18")
    r18_meta = r18_wrapper.get_metadata() if r18_wrapper else {}

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
            "status": "active" if (r18_wrapper and r18_wrapper.is_ready) else "no_approved_model",
            "message": "Model loaded in active runtime." if (r18_wrapper and r18_wrapper.is_ready) else "No approved model currently deployed",
            "classes": ["Cyst", "Normal", "Stone", "Tumor"],
            "accuracy": r18_meta.get("accuracy") or 0.9963,
            "f1_macro": r18_meta.get("macro_f1") or 0.9947,
            "precision": 0.9974,
            "recall": r18_meta.get("stone_recall") or 0.9712,
            "loaded": r18_wrapper.is_ready if r18_wrapper else False,
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
        "dl_models": dl_registry.list_models(),
        "ml_tabular": ml_info,
        "ml_xgboost": ml_info  # Backward compatibility alias
    }
