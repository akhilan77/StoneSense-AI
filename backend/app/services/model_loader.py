"""Model loader startup service.

Loads ML and DL models once at startup and stores them as singletons.
"""

from pathlib import Path
import logging
from typing import Dict, Any, Optional
import joblib

# Setup paths to import ML and DL training components
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "ml" / "training"))

logger = logging.getLogger("ModelLoader")


class ModelLoader:
    """Singleton model loader to load ML and DL models once."""
    
    _instance: Optional["ModelLoader"] = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelLoader, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ModuleNotFoundError:
            self.device = "cpu"
        self.dl_model = None
        self.ml_model = None
        self.ml_pipeline = None
        
        self._initialized = True

    def load_all_models(self) -> None:
        """Loads ResNet18 and XGBoost models once."""
        logger.info("Initializing StoneSense-AI Singleton Model Loader...")

        # 1. Load DL ResNet18 Model
        dl_model_path = PROJECT_ROOT / "dl" / "models" / "kidney_resnet18.pth"
        if dl_model_path.exists():
            try:
                import torch
                from model import build_resnet18_classifier

                logger.info(f"Loading ResNet18 weights from {dl_model_path}...")
                self.dl_model = build_resnet18_classifier(num_classes=4, freeze_backbone=False)
                self.dl_model.load_state_dict(torch.load(dl_model_path, map_location=self.device))
                self.dl_model.to(self.device)
                self.dl_model.eval()
                logger.info("ResNet18 loaded successfully.")
            except ModuleNotFoundError as exc:
                logger.error(f"DL dependencies unavailable; CT model not loaded: {exc}")
        else:
            logger.error(f"ResNet18 weights not found at {dl_model_path}")

        # 2. Load ML Risk Model & Pipelines
        ml_model_path = PROJECT_ROOT / "ml" / "models" / "kidney_risk_model.pkl"
        ml_pipeline_path = PROJECT_ROOT / "ml" / "artifacts" / "preprocessing_pipeline.pkl"
        
        if ml_model_path.exists() and ml_pipeline_path.exists():
            logger.info(f"Loading XGBoost model and pipeline...")
            self.ml_model = joblib.load(ml_model_path)
            self.ml_pipeline = joblib.load(ml_pipeline_path)
            logger.info("XGBoost and preprocessing pipeline loaded successfully.")
        else:
            logger.error("XGBoost or preprocessor pipeline artifacts missing.")


# Singleton helper access
model_loader = ModelLoader()
