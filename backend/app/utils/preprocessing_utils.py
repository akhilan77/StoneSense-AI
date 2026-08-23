"""Preprocessing utilities for mapping and structuring tabular clinical inputs."""

import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger("PreprocessingUtils")


def prepare_tabular_inputs(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """Standardizes dictionary keys to match expected ML pipeline training headers.

    Args:
        patient_data: Dict containing raw patient parameters.

    Returns:
        pd.DataFrame: Formatted DataFrame ready for ColumnTransformer pipeline.
    """
    feature_mapping = {
        "urine_specific_gravity": "gravity",
        "urine_ph": "ph",
        "calcium": "calc",
        "osmolality": "osmo",
        "conductivity": "cond",
    }

    mapped_features = {}
    for k, v in patient_data.items():
        mapped_key = feature_mapping.get(k, k)
        mapped_features[mapped_key] = v

    expected_cols = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
    return pd.DataFrame([{col: mapped_features.get(col, 0.0) for col in expected_cols}])
