"""Inference wrapper for Kidney Stone Risk prediction (ML).

Loads pre-fitted preprocessing pipeline and trained classifier to predict risk
probabilities from raw patient clinical urine chemistry records.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger("MLRiskInference")


class MLRiskInferenceEngine:
    """Inference engine for predicting kidney stone risk from urine analysis data."""

    def __init__(self, model_path: Path, pipeline_path: Path, class_mapping_path: Path):
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path)
        self.class_mapping_path = Path(class_mapping_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model pkl not found at {self.model_path}")
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline pkl not found at {self.pipeline_path}")
        if not self.class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping not found at {self.class_mapping_path}")

        # Load artifacts
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)
        
        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            self.class_mapping = json.load(f)

        logger.info("MLRiskInferenceEngine loaded successfully.")

    def predict_risk(self, clinical_data: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """Predicts risk probability and classification category for clinical inputs.
        
        Args:
            clinical_data: Clinical measurements (gravity, ph, osmo, cond, urea, calc).
            
        Returns:
            Dict: Predicted risk percentage and classification labels.
        """
        df = pd.DataFrame([clinical_data])

        # Preprocess features
        X_trans = self.pipeline.transform(df)

        # Predict
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        pred_label = int(self.model.predict(X_trans)[0])
        
        class_name = self.class_mapping[str(pred_label)]

        return {
            "kidney_stone_risk_probability": round(prob, 4),
            "predicted_risk_category": class_name,
            "risk_percentage": f"{prob * 100:.2f}%"
        }
