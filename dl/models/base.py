"""Base abstraction and contract interface for StoneSense-AI CT Image Models.

Provides unified prediction, explainability, and metadata interfaces across
diverse architectures: CNNs (ResNet18), YOLO (YOLO26), Vision Transformers (DINOv3),
and Quantum Classifiers (QKNN).
"""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging

logger = logging.getLogger("BaseCTModel")

CLASS_NAMES: List[str] = ["Cyst", "Normal", "Stone", "Tumor"]
CLASS_MAPPING: Dict[int, str] = {i: c for i, c in enumerate(CLASS_NAMES)}


class DLModelStatus(str, Enum):
    """Lifecycle and availability status for deep learning models."""
    READY = "READY"
    PENDING_WEIGHTS = "PENDING_WEIGHTS"
    ERROR = "ERROR"


class BaseCTModel(ABC):
    """Abstract base class representing a deep learning model for CT kidney stone classification."""

    def __init__(
        self,
        model_id: str,
        model_name: str,
        model_family: str,
        artifact_dir: Optional[Path] = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.model_family = model_family
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self._status = DLModelStatus.PENDING_WEIGHTS
        self._error_message: Optional[str] = None
        self._metrics: Dict[str, Any] = {}
        self._model_card: Dict[str, Any] = {}

    @property
    def status(self) -> DLModelStatus:
        """Current operational status of the model wrapper."""
        return self._status

    @property
    def error_message(self) -> Optional[str]:
        """Current error or pending reason message."""
        return self._error_message

    @property
    def is_ready(self) -> bool:
        """True if model artifacts exist and weights are loaded into memory."""
        return self._status == DLModelStatus.READY

    @abstractmethod
    def load(self) -> bool:
        """Loads weights, config, and required dependencies into memory.
        
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        pass

    @abstractmethod
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs inference on raw CT image bytes and returns structured predictions.
        
        Args:
            image_bytes: Raw binary bytes of uploaded CT scan.
            
        Returns:
            Dict containing:
                - model_id: str
                - model_name: str
                - class_name: str ("Cyst" | "Normal" | "Stone" | "Tumor")
                - confidence: float (0.0 - 1.0)
                - probabilities: Dict[str, float]
                - inference_time_sec: float
                - explainability: Dict[str, Any]
        """
        pass

    def explain(
        self,
        image_bytes: bytes,
        target_class: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates visual explanation (e.g., Grad-CAM) if supported by the architecture.
        
        Default implementation returns unsupported explanation safely without throwing errors.
        """
        return {
            "available": False,
            "overlay_url": "",
            "target_class": target_class or "Unknown",
            "message": f"Explainability visualization is not supported for {self.model_name}."
        }

    def load_metrics_artifact(self, metrics_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Loads evaluation metrics from metrics.json if available."""
        if metrics_path is None and self.artifact_dir:
            metrics_path = self.artifact_dir / "metrics.json"

        if metrics_path and Path(metrics_path).exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
                    return self._metrics
            except Exception as e:
                logger.warning(f"Could not load metrics from {metrics_path}: {e}")
        return {}

    def load_model_card_artifact(self, card_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Loads model card metadata from model_card.json if available."""
        if card_path is None and self.artifact_dir:
            card_path = self.artifact_dir / "model_card.json"

        if card_path and Path(card_path).exists():
            try:
                with open(card_path, "r", encoding="utf-8") as f:
                    self._model_card = json.load(f)
                    return self._model_card
            except Exception as e:
                logger.warning(f"Could not load model card from {card_path}: {e}")
        return {}

    def get_metadata(self) -> Dict[str, Any]:
        """Returns standard metadata dictionary for UI inspection and API responses."""
        acc = (
            self._metrics.get("accuracy")
            or self._metrics.get("test_accuracy")
            or self._metrics.get("test", {}).get("slice_accuracy")
        )
        macro_f1 = (
            self._metrics.get("f1_macro")
            or self._metrics.get("test_f1_macro")
            or self._metrics.get("test", {}).get("slice_macro_f1")
            or self._metrics.get("macro_f1")
        )
        
        per_class = self._metrics.get("per_class_metrics") or self._metrics.get("test", {}).get("per_class", {})
        stone_rec = None
        tumor_rec = None
        if isinstance(per_class, dict):
            stone_rec = per_class.get("Stone", {}).get("recall") or self._metrics.get("stone_recall")
            tumor_rec = per_class.get("Tumor", {}).get("recall") or self._metrics.get("tumor_recall")
        else:
            stone_rec = self._metrics.get("stone_recall")
            tumor_rec = self._metrics.get("tumor_recall")

        return {
            "id": self.model_id,
            "name": self.model_name,
            "family": self.model_family,
            "status": self._status.value,
            "is_ready": self.is_ready,
            "error_message": self._error_message,
            "accuracy": float(acc) if acc is not None else None,
            "macro_f1": float(macro_f1) if macro_f1 is not None else None,
            "stone_recall": float(stone_rec) if stone_rec is not None else None,
            "tumor_recall": float(tumor_rec) if tumor_rec is not None else None,
            "classes": CLASS_NAMES,
            "created_at": self._model_card.get("created_at"),
            "version_tag": self._model_card.get("version_tag", f"{self.model_id}_v1"),
            "display_name": self._model_card.get("display_name", self.model_name),
        }
