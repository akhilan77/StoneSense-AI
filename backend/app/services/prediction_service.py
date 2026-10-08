"""Prediction orchestration service layer.

Leverages singleton loaded model states and DLModelRegistry to perform
thread-safe inference on image and tabular patient inputs.
"""

import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "dl") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "dl"))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from app.services.model_loader import model_loader
from app.utils.preprocessing_utils import prepare_tabular_inputs

logger = logging.getLogger("PredictionService")


class PredictionService:
    """FastAPI Prediction Engine pulling pre-loaded models from ModelLoader and DLModelRegistry."""

    @staticmethod
    def predict_ct_image(image_bytes: bytes, model_id: Optional[str] = "resnet18") -> Dict[str, Any]:
        """Classifies CT scan image using the specified DL model wrapper.

        Args:
            image_bytes: Raw binary uploaded image bytes.
            model_id: Target model identifier ('resnet18', 'yolo26', 'dinov3', 'qknn'). Defaults to 'resnet18'.

        Returns:
            Dict: Unified predicted class, confidence float, model metadata, and probabilities.
        """
        from dl.models.registry import dl_registry

        target_id = model_id or "resnet18"
        try:
            model_wrapper = dl_registry.get_model(target_id)
        except KeyError as exc:
            raise RuntimeError(str(exc))

        if not model_wrapper.is_ready:
            # Fallback for resnet18 if legacy model_loader holds a raw model
            if target_id in ("resnet18", "resnet18_ct") and model_loader.dl_model is not None:
                from app.utils.image_utils import preprocess_ct_image
                from dl.training.model import CLASS_MAPPING as DL_CLASS_MAPPING
                from dl.preprocessing.transforms import get_val_test_transforms
                import torch

                val_tf = get_val_test_transforms(image_size=(224, 224))
                input_tensor = preprocess_ct_image(image_bytes, val_tf).to(model_loader.device)

                with torch.no_grad():
                    output = model_loader.dl_model(input_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred_idx = int(torch.argmax(output, dim=1).item())
                    confidence = float(probs[pred_idx].item())

                predicted_class = DL_CLASS_MAPPING[pred_idx]
                return {
                    "class": predicted_class,
                    "class_name": predicted_class,
                    "confidence": round(confidence, 4),
                    "model_id": "resnet18",
                    "model_name": "ResNet18",
                    "model_family": "resnet18_ct",
                    "probabilities": {DL_CLASS_MAPPING[i]: float(probs[i]) for i in range(len(DL_CLASS_MAPPING))},
                }
            raise RuntimeError(
                f"DL model '{target_id}' is not ready for inference "
                f"({model_wrapper._error_message or model_wrapper.status.value})."
            )

        res = model_wrapper.predict(image_bytes)
        
        # Standardize return dictionary
        return {
            "class": res["class_name"],
            "class_name": res["class_name"],
            "confidence": res["confidence"],
            "model_id": res["model_id"],
            "model_name": res["model_name"],
            "model_family": res["model_family"],
            "probabilities": res.get("probabilities", {}),
            "inference_time_sec": res.get("inference_time_sec", 0.0),
        }

    @staticmethod
    def predict_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates risk level probability using pre-loaded ML tabular model.

        Args:
            patient_data: Dict containing patient demographics/clinical parameters.

        Returns:
            Dict: Probability score and risk category label.
        """
        if model_loader.ml_model is None or model_loader.ml_pipeline is None:
            raise RuntimeError("ML Risk model or pipeline is not loaded.")

        # Prepare and preprocess tabular features
        df_input = prepare_tabular_inputs(patient_data)
        X_trans = model_loader.ml_pipeline.transform(df_input)

        # Predict
        prob = float(model_loader.ml_model.predict_proba(X_trans)[0, 1])
        pred_label = int(model_loader.ml_model.predict(X_trans)[0])
        risk_level = "High" if pred_label == 1 else "Low"

        logger.info(f"Clinical risk assessment: {risk_level} (Prob: {prob:.4f})")
        return {
            "probability": round(prob, 4),
            "risk": risk_level
        }

    @staticmethod
    def predict_complete(
        image_bytes: bytes,
        patient_data: Dict[str, Any],
        model_id: Optional[str] = "resnet18"
    ) -> Dict[str, Any]:
        """Runs both independent model streams without combining their probabilities."""
        ct_res = PredictionService.predict_ct_image(image_bytes, model_id=model_id)
        risk_res = PredictionService.predict_risk(patient_data)
        return {
            "ct_prediction": ct_res,
            "risk_prediction": risk_res
        }
