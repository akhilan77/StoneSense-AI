"""FastAPI health check router returning status of loaded singletons and system parameters."""

import sys
from fastapi import APIRouter

from app.services.model_loader import model_loader

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def get_health():
    """Returns application status, version, and model state verification."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": {
            "resnet18_ct_classification": model_loader.dl_model is not None,
            "xgboost_risk_prediction": model_loader.ml_model is not None,
            "preprocessing_pipeline": model_loader.ml_pipeline is not None
        },
        "hardware": {
            "cuda_available": str(model_loader.device).startswith("cuda"),
            "active_device": str(model_loader.device)
        }
    }
