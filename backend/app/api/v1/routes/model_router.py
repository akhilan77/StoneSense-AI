"""FastAPI router providing metadata specifications of trained models."""

from fastapi import APIRouter
from app.services.model_loader import model_loader
from pathlib import Path
import json

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def get_models_info():
    """Returns training parameters and validation statistics of current active models."""
    project_root = Path(__file__).resolve().parents[5]
    dl_metrics = json.loads((project_root / "dl" / "models" / "metrics.json").read_text(encoding="utf-8"))
    ml_metrics = json.loads((project_root / "ml" / "models" / "metrics.json").read_text(encoding="utf-8"))
    return {
        "dl_resnet18": {
            "name": "CT-KIDNEY-CLASSIFIER (ResNet18)",
            "classes": ["Cyst", "Normal", "Stone", "Tumor"],
            "accuracy": dl_metrics["accuracy"],
            "f1_macro": dl_metrics["f1_macro"],
            "target_layer": "model.FC / model.layer4[-1] for Grad-CAM",
            "loaded": model_loader.dl_model is not None
        },
        "ml_xgboost": {
            "name": "Urine-Analysis-Risk-Predictor (XGBoost)",
            "features": ["gravity", "ph", "osmo", "cond", "urea", "calc"],
            "validation_accuracy": ml_metrics["accuracy"],
            "validation_f1": ml_metrics["f1_score"],
            "validation_mcc": ml_metrics["matthews_correlation_coefficient"],
            "loaded": model_loader.ml_model is not None
        }
    }
