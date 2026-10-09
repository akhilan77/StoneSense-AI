"""DINOv3 Model Wrapper for CT Kidney Classification.

Encapsulates Meta DINOv3 Vision Transformer backbone feature extraction and
trained champion classification head.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import io
import time
import logging
import joblib
import numpy as np
from PIL import Image

from dl.models.base import BaseCTModel, DLModelStatus, CLASS_MAPPING, CLASS_NAMES

logger = logging.getLogger("DINOv3Wrapper")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DINOv3Wrapper(BaseCTModel):
    """Wrapper for DINOv3 Vision Foundation Model + Trained Classifier Head."""

    def __init__(
        self,
        artifact_dir: Optional[Path] = None,
        backbone_dir: Optional[Path] = None,
    ):
        super().__init__(
            model_id="dinov3",
            model_name="DINOv3",
            model_family="dinov3_ct",
            artifact_dir=artifact_dir or (PROJECT_ROOT / "dl" / "models" / "ct" / "dinov3"),
        )
        self.backbone_dir = backbone_dir or (self.artifact_dir / "backbone" if self.artifact_dir else None)
        self.backbone = None
        self.processor = None
        self.classifier_head = None
        self.config = None

    def load(self) -> bool:
        """Attempts to load DINOv3 classifier head and backbone if artifacts exist."""
        self.load_metrics_artifact()
        self.load_model_card_artifact()

        if not self.artifact_dir or not self.artifact_dir.exists():
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "DINOv3 artifact directory not found. Waiting for training export."
            return False

        head_path = self.artifact_dir / "champion_head.pkl"
        cfg_path = self.artifact_dir / "dino_backbone_config.json"

        if not head_path.exists():
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = f"DINOv3 head checkpoint missing at {head_path}."
            return False

        # Load trained MLP champion head
        try:
            self.classifier_head = joblib.load(head_path)
        except Exception as e:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = f"Failed to load DINOv3 head checkpoint: {e}"
            return False

        # Determine backbone identifier / path
        model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        if cfg_path.exists():
            import json
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                    model_name = self.config.get("model_id") or self.config.get("model_name", model_name)
            except Exception as e:
                logger.warning(f"Could not parse dino_backbone_config.json: {e}")

        # Check for local backbone directory to avoid unauthenticated/offline HF downloads
        env_backbone = os.environ.get("DINOV3_BACKBONE_DIR")
        target_backbone_path = None
        if env_backbone and Path(env_backbone).exists():
            target_backbone_path = Path(env_backbone)
        elif self.backbone_dir and self.backbone_dir.exists():
            target_backbone_path = self.backbone_dir
        elif (self.artifact_dir / "dinov3_backbone").exists():
            target_backbone_path = self.artifact_dir / "dinov3_backbone"

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            if target_backbone_path:
                logger.info(f"Loading DINOv3 backbone locally from {target_backbone_path}")
                self.processor = AutoImageProcessor.from_pretrained(str(target_backbone_path), local_files_only=True)
                self.backbone = AutoModel.from_pretrained(str(target_backbone_path), local_files_only=True)
                self.backbone.eval()
                self._status = DLModelStatus.READY
                self._error_message = None
                logger.info("DINOv3Wrapper loaded successfully from local backbone.")
                return True
            else:
                # In production/air-gapped environments without local backbone files,
                # mark status as PENDING_WEIGHTS rather than failing or blocking
                self._status = DLModelStatus.PENDING_WEIGHTS
                self._error_message = (
                    f"DINOv3 backbone weights missing locally (expected at '{self.artifact_dir / 'backbone'}' "
                    f"or via DINOV3_BACKBONE_DIR). Classifier head is verified."
                )
                logger.info(self._error_message)
                return False
        except ImportError:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "transformers/torch dependencies missing for DINOv3."
            return False
        except Exception as exc:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = f"DINOv3 backbone loading pending: {exc}"
            return False

    def predict_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Runs classifier head inference directly on a 384-dimensional DINOv3 feature vector."""
        if self.classifier_head is None:
            head_path = self.artifact_dir / "champion_head.pkl" if self.artifact_dir else None
            if head_path and head_path.exists():
                self.classifier_head = joblib.load(head_path)
            else:
                raise RuntimeError("DINOv3 classifier head is not loaded.")

        start_t = time.perf_counter()
        emb_2d = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        probs = self.classifier_head.predict_proba(emb_2d)[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_MAPPING.get(pred_idx, "Unknown")
        confidence = float(probs[pred_idx])

        elapsed = time.perf_counter() - start_t
        probabilities = {CLASS_MAPPING[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "class_name": predicted_class,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "inference_time_sec": round(elapsed, 4),
        }

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extracts DINOv3 CLS embeddings and executes classifier head forward pass."""
        if not self.is_ready or self.backbone is None or self.processor is None or self.classifier_head is None:
            raise RuntimeError(f"DINOv3 model is not available ({self._error_message}).")

        import torch

        start_t = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = self.backbone(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output.cpu().numpy()
            else:
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()

        probs = self.classifier_head.predict_proba(embedding)[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = CLASS_MAPPING.get(pred_idx, "Unknown")
        confidence = float(probs[pred_idx])

        elapsed = time.perf_counter() - start_t
        probabilities = {CLASS_MAPPING[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "class_name": predicted_class,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "inference_time_sec": round(elapsed, 4),
        }

    def explain(
        self,
        image_bytes: bytes,
        target_class: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Safe fallback: Explainability is currently unavailable for DINOv3."""
        return {
            "available": False,
            "overlay_url": "",
            "target_class": target_class or "Unknown",
            "message": "Explainability is not currently available for DINOv3."
        }
