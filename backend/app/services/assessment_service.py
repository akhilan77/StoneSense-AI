"""Assessment orchestration service layer (Phase 7A).

Aggregates diagnostic prediction engines and explainability services
to return a unified final patient assessment.
"""

from datetime import datetime
import time
from typing import Dict, Any, Optional
import pandas as pd

from app.services.prediction_service import PredictionService
from app.services.model_loader import model_loader
from app.utils.preprocessing_utils import prepare_tabular_inputs

# Configure paths for explainability engines
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "dl" / "explainability"))


class AssessmentService:
    """Orchestrator to aggregate model inferences and explainability overlays."""

    @staticmethod
    def build_assessment(
        patient_data: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        gradcam_out_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs predictions and explanations to construct a complete unified patient report.
        
        Args:
            patient_data: Tabular clinical parameters dict.
            image_bytes: Optional binary CT scan scan bytes.
            gradcam_out_path: Optional file path to save the generated Grad-CAM overlay.
            
        Returns:
            Dict: Unified structured assessment payload.
        """
        start_time = time.time()

        # 1. Tabular Risk prediction & SHAP explainability
        risk_prediction = PredictionService.predict_risk(patient_data)
        
        # Calculate SHAP values
        shap_values_dict = {}
        if model_loader.ml_model is not None and model_loader.ml_pipeline is not None:
            try:
                import shap
            except ModuleNotFoundError:
                shap = None

            if shap is not None:
                df_input = prepare_tabular_inputs(patient_data)
                X_trans = model_loader.ml_pipeline.transform(df_input)
                expected_cols = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
                df_trans = pd.DataFrame(X_trans, columns=expected_cols)
                
                explainer = shap.TreeExplainer(model_loader.ml_model)
                shap_explanation = explainer(df_trans)[0]
                
                for col, val in zip(expected_cols, shap_explanation.values):
                    shap_values_dict[col] = float(val)

        # 2. CT classification & Grad-CAM visual activation overlay
        ct_prediction = {}
        gradcam_res = {}
        
        if image_bytes is not None and model_loader.dl_model is not None:
            ct_prediction = PredictionService.predict_ct_image(image_bytes)
            
            # If a local saving path is provided, run Grad-CAM and write image
            if gradcam_out_path:
                from PIL import Image
                import io
                import numpy as np
                import cv2
                from pytorch_grad_cam.utils.image import show_cam_on_image
                from transforms import get_val_test_transforms

                device = model_loader.device
                val_tf = get_val_test_transforms(image_size=(224, 224))
                
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_np = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
                input_tensor = val_tf(img_pil).unsqueeze(0).to(device)

                # Initialize explainer target
                cam_engine = shap_explanation = None # reuse imports
                from pytorch_grad_cam import GradCAM
                cam = GradCAM(model=model_loader.dl_model, target_layers=[model_loader.dl_model.layer4[-1]])
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                out_p = Path(gradcam_out_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_p), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                gradcam_res = {"overlay_path": str(out_p)}

        elapsed_time = time.time() - start_time

        # Generate recommendation
        calc_val = patient_data.get("calcium", 0.0)
        risk_level = risk_prediction.get("risk", "Low")
        
        if risk_level == "High" or calc_val > 5.0:
            rec = "Elevated urinary calcium detected. Increase daily fluid intake, restrict dietary sodium, and schedule a specialist evaluation."
        else:
            rec = "Clinical parameters are within normal thresholds. Maintain standard hydration and routine checkups."

        return {
            "ct_prediction": ct_prediction,
            "risk_prediction": risk_prediction,
            "gradcam": gradcam_res,
            "shap": {
                "top_features": [
                    k for k, v in sorted(shap_values_dict.items(), key=lambda item: abs(item[1]), reverse=True)
                ],
                "feature_contributions": shap_values_dict
            },
            "recommendation": rec,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_sec": round(elapsed_time, 4)
            }
        }
