"""YOLO26 Model Wrapper for CT Kidney Classification.

Encapsulates Ultralytics YOLO26 classification fine-tuned model inference.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import io
import time
import logging
import numpy as np
from PIL import Image

from dl.models.base import BaseCTModel, DLModelStatus, CLASS_MAPPING, CLASS_NAMES

logger = logging.getLogger("YOLO26Wrapper")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class YOLO26Wrapper(BaseCTModel):
    """Wrapper for Ultralytics YOLO26 CT classification model."""

    def __init__(
        self,
        artifact_dir: Optional[Path] = None,
        weights_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            model_id="yolo26",
            model_name="YOLO26",
            model_family="yolo26_ct",
            artifact_dir=artifact_dir or (PROJECT_ROOT / "dl" / "models" / "ct" / "yolo26"),
        )
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None

    def load(self) -> bool:
        """Loads Ultralytics YOLO model from best.pt checkpoint if available."""
        # Always attempt to load metadata/metrics even if weights are pending
        self.load_metrics_artifact()
        self.load_model_card_artifact()

        target_weights = None
        if self.weights_path and self.weights_path.exists():
            target_weights = self.weights_path
        elif self.artifact_dir:
            p = self.artifact_dir / "best.pt"
            if p.exists():
                target_weights = p

        if not target_weights:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "YOLO26 weights file missing (expected dl/models/ct/yolo26/best.pt)"
            logger.info(self._error_message)
            return False

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(target_weights))
            self._status = DLModelStatus.READY
            self._error_message = None
            logger.info(f"YOLO26Wrapper loaded successfully from {target_weights}")
            return True
        except ImportError:
            self._status = DLModelStatus.PENDING_WEIGHTS
            self._error_message = "ultralytics package is not installed."
            logger.warning(self._error_message)
            return False
        except Exception as exc:
            self._status = DLModelStatus.ERROR
            self._error_message = f"Failed to load YOLO26 weights: {exc}"
            logger.exception(self._error_message)
            return False

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs classification inference on image bytes via Ultralytics."""
        if not self.is_ready or self.model is None:
            raise RuntimeError(f"YOLO26 is not ready for inference ({self._error_message}).")

        start_t = time.perf_counter()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self.model.predict(img, verbose=False)
        first_res = results[0]

        if hasattr(first_res, "probs") and first_res.probs is not None:
            probs_tensor = first_res.probs.data.cpu().numpy()
            pred_idx = int(first_res.probs.top1)
            confidence = float(first_res.probs.top1conf)
            probabilities = {
                CLASS_MAPPING.get(i, f"Class_{i}"): float(probs_tensor[i])
                for i in range(len(probs_tensor))
            }
            predicted_class = CLASS_MAPPING.get(pred_idx, first_res.names.get(pred_idx, "Unknown"))
        else:
            raise RuntimeError("YOLO26 model output did not contain classification probabilities.")

        elapsed = time.perf_counter() - start_t

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
        """Safe fallback: Explainability is currently not implemented for YOLO26."""
        return {
            "available": False,
            "overlay_url": "",
            "target_class": target_class or "Unknown",
            "message": "Explainability visualization is not implemented for YOLO26."
        }
