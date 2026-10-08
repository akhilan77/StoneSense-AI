"""Model loader startup service.

Loads the centralized ML model and DL federated model once at startup and stores
them as separate singletons. Supports DL checkpoint reloading when rounds complete.
"""

from pathlib import Path
import logging
from typing import Dict, Any, Optional
import joblib

# Setup paths to import ML and DL training components
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))
if str(PROJECT_ROOT / "dl" / "preprocessing") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
if str(PROJECT_ROOT / "ml" / "training") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "ml" / "training"))
if str(PROJECT_ROOT / "ml" / "models") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "ml" / "models"))

logger = logging.getLogger("ModelLoader")


class ModelLoader:
    """Singleton loader for the centralized ML and DL federated models."""

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
        self.active_dl_version_tag = "resnet18_centralized_v1"
        self.active_ml_version_tag = "logistic_regression_v001"
        self.active_ml_family = "logistic_regression"

        self._initialized = True

    def load_all_models(self) -> None:
        """Loads the DL federated ResNet18 and centralized tabular risk models."""
        logger.info("Initializing StoneSense-AI Singleton Model Loader...")
        self.reload_dl_model()
        self.load_ml_model()

    def reload_dl_model(self, version_tag: Optional[str] = None) -> bool:
        """Dynamically loads or reloads the active ResNet18 DL model."""
        import torch
        from dl.training.model import build_resnet18_classifier

        fed_dir = PROJECT_ROOT / "dl" / "models" / "federated"
        target_path = None

        if version_tag:
            custom_path = fed_dir / f"{version_tag}.pth"
            if custom_path.exists():
                target_path = custom_path
                self.active_dl_version_tag = version_tag

        if not target_path:
            latest_path = fed_dir / "latest.pth"
            if latest_path.exists():
                target_path = latest_path
                try:
                    data = torch.load(latest_path, map_location="cpu", weights_only=False)
                    self.active_dl_version_tag = data.get("version_tag", "resnet18_fed_latest")
                except Exception:
                    self.active_dl_version_tag = "resnet18_fed_latest"
            else:
                default_path = PROJECT_ROOT / "dl" / "models" / "kidney_resnet18.pth"
                if default_path.exists():
                    target_path = default_path
                    self.active_dl_version_tag = "resnet18_centralized_v1"

        if target_path and target_path.exists():
            try:
                logger.info(f"Loading ResNet18 weights from {target_path} (Version: {self.active_dl_version_tag})...")
                self.dl_model = build_resnet18_classifier(num_classes=4, freeze_backbone=False)
                checkpoint_data = torch.load(target_path, map_location=self.device, weights_only=False)

                if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                    self.dl_model.load_state_dict(checkpoint_data["model_state_dict"])
                else:
                    self.dl_model.load_state_dict(checkpoint_data)

                self.dl_model.to(self.device)
                self.dl_model.eval()
                logger.info(f"ResNet18 ({self.active_dl_version_tag}) loaded successfully.")
                return True
            except Exception as exc:
                logger.error(f"Error loading DL weights from {target_path}: {exc}")
                return False
        else:
            logger.warning(f"No ResNet18 checkpoint found at {target_path}")
            return False

    def load_ml_model(self, version_tag: Optional[str] = None) -> bool:
        """Loads the active tabular risk model from ModelRegistry and preprocessing pipeline."""
        ml_pipeline_path = PROJECT_ROOT / "ml" / "artifacts" / "preprocessing_pipeline.pkl"

        try:
            from ml.risk_models import registry
            active_model = registry.get_model(version_tag)
            self.ml_model = active_model
            self.active_ml_version_tag = active_model.version_tag
            self.active_ml_family = active_model.model_family

            if ml_pipeline_path.exists():
                self.ml_pipeline = joblib.load(ml_pipeline_path)
            else:
                logger.error(f"Preprocessing pipeline artifact missing at {ml_pipeline_path}")
                return False

            logger.info(
                f"Tabular Risk Model loaded successfully from registry "
                f"({self.active_ml_version_tag}, family: {self.active_ml_family})."
            )
            return True
        except Exception as exc:
            logger.error(f"Failed to load risk model from registry: {exc}")
            # Fallback to direct artifact load if registry fails unexpectedly
            ml_model_path = PROJECT_ROOT / "ml" / "models" / "candidate_risk_model.pkl"
            if not ml_model_path.exists():
                ml_model_path = PROJECT_ROOT / "ml" / "models" / "kidney_risk_model.pkl"
            if ml_model_path.exists() and ml_pipeline_path.exists():
                self.ml_model = joblib.load(ml_model_path)
                self.ml_pipeline = joblib.load(ml_pipeline_path)
                self.active_ml_version_tag = ml_model_path.stem
                logger.info(f"Fallback model loaded from {ml_model_path}")
                return True
            return False


# Singleton helper access
model_loader = ModelLoader()
