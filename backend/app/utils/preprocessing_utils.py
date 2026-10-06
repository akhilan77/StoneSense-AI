"""Preprocessing utilities for mapping and structuring tabular clinical inputs."""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger("PreprocessingUtils")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "ml" / "artifacts" / "feature_columns.pkl"


def get_expected_feature_columns() -> List[str]:
    """Retrieves authoritative feature columns from feature_columns.pkl."""
    if FEATURE_COLUMNS_PATH.exists():
        return list(joblib.load(FEATURE_COLUMNS_PATH))
    return ["gravity", "ph", "osmo", "cond", "urea", "calc"]


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

    expected_cols = get_expected_feature_columns()
    return pd.DataFrame([{col: mapped_features.get(col, 0.0) for col in expected_cols}])

