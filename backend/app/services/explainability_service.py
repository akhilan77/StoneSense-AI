"""Explainability service for CT classification (Grad-CAM) and ML risk (SHAP).

Calculates visual saliency maps and tabular feature importances for independent model outputs.
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import joblib

# Setup paths to import ML and DL training components
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "dl" / "explainability"))
sys.path.append(str(PROJECT_ROOT / "ml" / "training"))
sys.path.append(str(PROJECT_ROOT / "ml" / "explainability"))

from model import build_resnet18_classifier as build_resnet18, CLASS_MAPPING as DL_CLASS_MAPPING
from transforms import get_val_test_transforms
from pytorch_grad_cam import GradCAM
try:
    import shap
except ImportError:
    shap = None
import pandas as pd

logger = logging.getLogger("ExplainabilityService")


def generate_shap_for_patient(patient_features: Dict[str, Any]) -> Dict[str, Any]:
    """Return local SHAP values using the models already loaded for inference."""
    from app.services.model_loader import model_loader
    from app.utils.preprocessing_utils import prepare_tabular_inputs

    if model_loader.ml_model is None or model_loader.ml_pipeline is None:
        raise RuntimeError("ML model or preprocessing pipeline is not loaded.")
    frame = prepare_tabular_inputs(patient_features)
    transformed = model_loader.ml_pipeline.transform(frame)
    columns = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
    explanation = shap.TreeExplainer(model_loader.ml_model)(pd.DataFrame(transformed, columns=columns))[0]
    contributions = {name: float(value) for name, value in zip(columns, explanation.values)}
    return {
        "top_features": sorted(contributions, key=lambda name: abs(contributions[name]), reverse=True),
        "feature_contributions": contributions,
        "feature_directions": {
            name: "increases" if value > 0 else "decreases" if value < 0 else "neutral"
            for name, value in contributions.items()
        },
        "summary": "The strongest SHAP contributors influenced the model output for this request; they do not establish causation.",
    }


def generate_gradcam_for_bytes(image_bytes: bytes, output_path: str, target_class: Optional[str] = None) -> Dict[str, Any]:
    """Generate a class-specific Grad-CAM overlay for an uploaded CT image."""
    from app.services.model_loader import model_loader
    from app.utils.image_utils import preprocess_ct_image
    import io
    import cv2
    import torch

    if model_loader.dl_model is None:
        raise RuntimeError("DL model is not loaded.")

    model = model_loader.dl_model
    model.eval()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_rgb = np.asarray(image, dtype=np.uint8)
    transform = get_val_test_transforms(image_size=(224, 224))
    tensor = preprocess_ct_image(image_bytes, transform).to(model_loader.device)
    tensor.requires_grad_(True)

    activations = []
    gradients = []

    target_layer = getattr(model.layer4[-1], "conv2", model.layer4[-1])

    def _forward_hook(module, inputs, output):
        activations.append(output.detach())

    def _backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(_forward_hook)
    backward_handle = target_layer.register_full_backward_hook(_backward_hook)

    try:
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_label = DL_CLASS_MAPPING.get(pred_idx, "Normal")
        target_idx = pred_idx
        if target_class is not None:
            for idx, label in DL_CLASS_MAPPING.items():
                if label.lower() == str(target_class).lower():
                    target_idx = idx
                    break
        if pred_label == "Normal":
            target_idx = pred_idx

        model.zero_grad(set_to_none=True)
        target_score = logits[:, target_idx].sum()
        target_score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

    act = activations[-1]
    grad = gradients[-1]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam[0, 0].detach().cpu()
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    else:
        cam = torch.zeros_like(cam)

    cam_np = cam.numpy()
    heatmap_resized = cv2.resize(cam_np, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    original_rgb_float = original_rgb.astype(np.float32) / 255.0
    overlay = np.clip(0.5 * original_rgb_float + 0.5 * heatmap_rgb, 0.0, 1.0)
    overlay_u8 = (overlay * 255).astype(np.uint8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output), cv2.cvtColor(overlay_u8, cv2.COLOR_RGB2BGR))
    if not saved:
        raise RuntimeError(f"Failed to save Grad-CAM overlay to {output}")

    result = {
        "overlay_path": str(output),
        "target_class": DL_CLASS_MAPPING.get(target_idx, pred_label),
        "prediction": pred_label,
        "confidence": round(float(probs[0, target_idx].item()), 4),
        "available": pred_label != "Normal",
        "message": (
            "No stone-specific localization is shown because the model classified this scan as Normal."
            if pred_label == "Normal"
            else "The highlighted regions indicate areas that contributed to the model's prediction."
        )
    }
    return result


class ExplainabilityService:
    """Unified explainability manager exposing Grad-CAM and SHAP interface wrappers."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load DL ResNet18 model
        dl_model_path = PROJECT_ROOT / "dl" / "models" / "kidney_resnet18.pth"
        if dl_model_path.exists():
            self.dl_model = build_resnet18(num_classes=4, freeze_backbone=False)
            self.dl_model.load_state_dict(torch.load(dl_model_path, map_location=self.device))
            self.dl_model.to(self.device)
            self.dl_model.eval()
            self.dl_cam = GradCAM(model=self.dl_model, target_layers=[self.dl_model.layer4[-1]])
            self.dl_transform = get_val_test_transforms(image_size=(224, 224))
            logger.info("DL Grad-CAM model loaded in ExplainabilityService.")
        else:
            self.dl_model = None
            logger.warning(f"DL model checkpoint not found at {dl_model_path}")

        # Load ML Risk model and pipeline
        ml_model_path = PROJECT_ROOT / "ml" / "models" / "kidney_risk_model.pkl"
        ml_pipeline_path = PROJECT_ROOT / "ml" / "artifacts" / "preprocessing_pipeline.pkl"
        if ml_model_path.exists() and ml_pipeline_path.exists():
            self.ml_model = joblib.load(ml_model_path)
            self.ml_pipeline = joblib.load(ml_pipeline_path)
            self.ml_explainer = shap.TreeExplainer(self.ml_model)
            logger.info("ML SHAP Risk model loaded in ExplainabilityService.")
        else:
            self.ml_model = None
            logger.warning("ML model or preprocessor artifacts missing.")

    def generate_gradcam(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generates Grad-CAM activation heatmap overlay.

        Args:
            image_path: Absolute or relative file path to the input CT scan.
            output_path: Optional file path to save the generated overlay.

        Returns:
            Dict: Classification prediction, confidence score, and saved overlay path.
        """
        if self.dl_model is None:
            return {"error": "DL ResNet18 model not loaded in service."}

        img_path = Path(image_path)
        if not img_path.exists():
            return {"error": f"Image file not found: {image_path}"}

        # Load image
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb.resize((224, 224)), dtype=np.float32) / 255.0
            input_tensor = self.dl_transform(img_rgb).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.dl_model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = int(torch.argmax(output, dim=1).item())
            confidence = float(probs[pred_idx].item())

        predicted_class = DL_CLASS_MAPPING[pred_idx]

        # Generate CAM
        grayscale_cam = self.dl_cam(input_tensor=input_tensor, targets=None)[0, :]
        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        # Save if requested
        saved_path = ""
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            # Save overlay image
            import cv2
            cv2.imwrite(str(out_p), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved_path = str(out_p)

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "gradcam_image_path": saved_path
        }

    def generate_shap(self, patient_features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates SHAP feature contribution attributions for clinical measurements.

        Args:
            patient_features: Dict containing gravity, ph, osmo, cond, urea, calc features.

        Returns:
            Dict: Predicted risk probability, level classification, and SHAP contributions.
        """
        if self.ml_model is None:
            return {"error": "ML Risk model or preprocessor not loaded in service."}

        # Ensure correct key formats
        feature_mapping = {
            "urine_specific_gravity": "gravity",
            "urine_ph": "ph",
            "calcium": "calc"
        }
        mapped_features = {}
        for k, v in patient_features.items():
            mapped_key = feature_mapping.get(k, k)
            mapped_features[mapped_key] = v

        # Filter features expected by the pipeline
        expected_cols = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
        df_input = pd.DataFrame([{col: mapped_features.get(col, 0.0) for col in expected_cols}])

        # Transform features
        X_trans = self.ml_pipeline.transform(df_input)

        # Predict
        prob = float(self.ml_model.predict_proba(X_trans)[0, 1])
        pred_label = int(self.ml_model.predict(X_trans)[0])
        risk_level = "high" if pred_label == 1 else "low"

        # Calculate SHAP values
        df_trans = pd.DataFrame(X_trans, columns=expected_cols)
        shap_explanation = self.ml_explainer(df_trans)[0]

        # Extract values
        shap_values_dict = {}
        for col, val in zip(expected_cols, shap_explanation.values):
            shap_values_dict[col] = float(val)

        return {
            "probability": round(prob, 4),
            "risk_level": risk_level,
            "shap_values": shap_values_dict
        }

    def generate_explanation(self, image_path: str, patient_features: Dict[str, Any], gradcam_out_path: Optional[str] = None) -> Dict[str, Any]:
        """Orchestrates combined diagnostics & explainability for CT scan and clinical features."""
        gradcam_res = self.generate_gradcam(image_path, gradcam_out_path)
        shap_res = self.generate_shap(patient_features)

        # Build clinical natural language summary
        ct_pred = gradcam_res.get("prediction", "N/A")
        risk_level = shap_res.get("risk_level", "N/A")

        explanation_text = (
            f"CT scan classification identifies kidney condition as '{ct_pred}'. "
            f"Urine biochemistry analysis assesses patient stone risk level as '{risk_level}' "
            f"with probability {shap_res.get('probability', 0.0) * 100:.2f}%."
        )

        return {
            "ct_prediction": ct_pred,
            "ct_confidence": gradcam_res.get("confidence", 0.0),
            "gradcam_image_path": gradcam_res.get("gradcam_image_path", ""),
            "risk_prediction": risk_level,
            "risk_probability": shap_res.get("probability", 0.0),
            "shap_feature_contributions": shap_res.get("shap_values", {}),
            "human_readable_explanation": {
                "summary": explanation_text,
                "dominant_urine_features": [
                    k for k, v in sorted(shap_res.get("shap_values", {}).items(), key=lambda item: abs(item[1]), reverse=True)
                ]
            }
        }
