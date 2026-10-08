"""Centralized DL Model Registry for StoneSense-AI.

Registers, loads, and manages all 4 CT image classification models:
- ResNet18 (PyTorch CNN)
- YOLO26 (Ultralytics YOLO26n-cls)
- DINOv3 (Meta Vision Transformer Foundation)
- QKNN (PennyLane Quantum K-Nearest Neighbors)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from dl.models.base import BaseCTModel, DLModelStatus
from dl.models.wrappers.resnet18_wrapper import ResNet18Wrapper
from dl.models.wrappers.yolo26_wrapper import YOLO26Wrapper
from dl.models.wrappers.dinov3_wrapper import DINOv3Wrapper
from dl.models.wrappers.qknn_wrapper import QKNNWrapper

logger = logging.getLogger("DLModelRegistry")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DLModelRegistry:
    """Registry coordinating discovery, initialization, and dispatching of CT DL models."""

    def __init__(self):
        self._models: Dict[str, BaseCTModel] = {}
        self._alias_map: Dict[str, str] = {
            "resnet18": "resnet18",
            "resnet18_ct": "resnet18",
            "resnet": "resnet18",
            "yolo26": "yolo26",
            "yolo26_ct": "yolo26",
            "yolo": "yolo26",
            "dinov3": "dinov3",
            "dinov3_ct": "dinov3",
            "dino": "dinov3",
            "qknn": "qknn",
            "qknn_ct": "qknn",
        }
        self._initialize_wrappers()

    def _initialize_wrappers(self) -> None:
        """Initializes model wrapper instances with their standard artifact directories."""
        ct_models_dir = PROJECT_ROOT / "dl" / "models" / "ct"
        
        self._models["resnet18"] = ResNet18Wrapper(
            artifact_dir=ct_models_dir / "resnet18",
            weights_path=PROJECT_ROOT / "dl" / "models" / "kidney_resnet18.pth",
        )
        self._models["yolo26"] = YOLO26Wrapper(
            artifact_dir=ct_models_dir / "yolo26",
        )
        self._models["dinov3"] = DINOv3Wrapper(
            artifact_dir=ct_models_dir / "dinov3",
        )
        self._models["qknn"] = QKNNWrapper(
            artifact_dir=ct_models_dir / "qknn",
        )

    def load_all_models(self) -> Dict[str, bool]:
        """Loads weights and dependencies for all models whose artifacts exist."""
        results = {}
        logger.info("Initializing DL Model Registry and loading available models...")
        for model_id, model in self._models.items():
            try:
                success = model.load()
                results[model_id] = success
                logger.info(f"Model '{model_id}' status: {model.status.value}")
            except Exception as e:
                logger.error(f"Error loading model '{model_id}': {e}")
                results[model_id] = False
        return results

    def resolve_model_id(self, identifier: Optional[str]) -> str:
        """Normalizes model ID or family alias to standard identifier."""
        if not identifier:
            return "resnet18"
        normalized = str(identifier).strip().lower()
        return self._alias_map.get(normalized, normalized)

    def get_model(self, identifier: Optional[str] = None) -> BaseCTModel:
        """Returns the model wrapper for the specified identifier (default: 'resnet18')."""
        model_id = self.resolve_model_id(identifier)
        if model_id not in self._models:
            raise KeyError(
                f"Unknown DL model identifier '{identifier}'. "
                f"Available models: {list(self._models.keys())}"
            )
        return self._models[model_id]

    def is_model_ready(self, identifier: str) -> bool:
        """Checks if the specified model is ready for inference."""
        model_id = self.resolve_model_id(identifier)
        if model_id not in self._models:
            return False
        return self._models[model_id].is_ready

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns metadata for all 4 models in standard order."""
        order = ["resnet18", "yolo26", "dinov3", "qknn"]
        return [self._models[mid].get_metadata() for mid in order if mid in self._models]


# Global Registry Singleton
dl_registry = DLModelRegistry()
