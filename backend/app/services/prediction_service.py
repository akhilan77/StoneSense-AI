"""Prediction orchestration service layer (Phase 7A).

Leverages singleton loaded model states to perform thread-safe inference
on image and tabular patient inputs. Exposes explainability hooks.
"""

import sys
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

# Include path references to explainability engines
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "ml" / "training"))

from app.services.model_loader import model_loader
from app.utils.image_utils import preprocess_ct_image
from app.utils.preprocessing_utils import prepare_tabular_inputs
from model import CLASS_MAPPING as DL_CLASS_MAPPING
from transforms import get_val_test_transforms

logger = logging.getLogger("PredictionService")


class PredictionService:
    """FastAPI Prediction Engine pulling pre-loaded models from ModelLoader."""

    @staticmethod
    def predict_ct_image(image_bytes: bytes) -> Dict[str, Any]:
        """Classifies CT scan image using ResNet18 model singleton.
        
        Args:
            image_bytes: Raw binary uploaded image bytes.
            
        Returns:
            Dict: Predicted class and confidence float.
        """
        if model_loader.dl_model is None:
            raise RuntimeError("DL ResNet18 model is not loaded in prediction service.")

        # Transform raw bytes to input tensor
        val_tf = get_val_test_transforms(image_size=(224, 224))
        input_tensor = preprocess_ct_image(image_bytes, val_tf).to(model_loader.device)

        # Run forward pass thread-safely
        with torch.no_grad():
            output = model_loader.dl_model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = int(torch.argmax(output, dim=1).item())
            confidence = float(probs[pred_idx].item())

        predicted_class = DL_CLASS_MAPPING[pred_idx]

        logger.info(f"CT classification result: {predicted_class} (Conf: {confidence:.4f})")
        return {
            "class": predicted_class,
            "confidence": round(confidence, 4)
        }

    @staticmethod
    def predict_risk(patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates risk level probability using pre-loaded XGBoost model.
        
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
    def predict_complete(image_bytes: bytes, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performs multi-modal prediction running both DL and ML models."""
        ct_res = PredictionService.predict_ct_image(image_bytes)
        risk_res = PredictionService.predict_risk(patient_data)
        return {
            "ct_prediction": ct_res,
            "risk_prediction": risk_res
        }
