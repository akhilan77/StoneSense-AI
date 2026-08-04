"""FastAPI router providing metadata specifications of trained models."""

from fastapi import APIRouter
from app.services.model_loader import model_loader

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def get_models_info():
    """Returns training parameters and validation statistics of current active models."""
    return {
        "dl_resnet18": {
            "name": "CT-KIDNEY-CLASSIFIER (ResNet18)",
            "classes": ["Cyst", "Normal", "Stone", "Tumor"],
            "accuracy": 0.9850,
            "f1_macro": 0.9793,
            "target_layer": "model.FC / model.layer4[-1] for Grad-CAM",
            "loaded": model_loader.dl_model is not None
        },
        "ml_xgboost": {
            "name": "Urine-Analysis-Risk-Predictor (XGBoost)",
            "features": ["gravity", "ph", "osmo", "cond", "urea", "calc"],
            "validation_accuracy": 0.9167,
            "validation_f1": 0.9091,
            "validation_mcc": 0.8452,
            "loaded": model_loader.ml_model is not None
        }
    }
