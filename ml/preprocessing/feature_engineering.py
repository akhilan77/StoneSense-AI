"""Feature Engineering utilities for StoneSense-AI (ML).

Extracts metadata, derives domain-specific physical/chemical feature ratios,
and constructs dataset metadata dictionaries.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np

logger = logging.getLogger("FeatureEngineering")


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives physiological ratio features for urine analysis if columns are available."""
    df_engineered = df.copy()

    # Calculate conductivity to osmolality ratio if both exist
    if 'cond' in df_engineered.columns and 'osmo' in df_engineered.columns:
        # Avoid division by zero
        df_engineered['cond_osmo_ratio'] = np.where(
            df_engineered['osmo'] != 0,
            df_engineered['cond'] / df_engineered['osmo'],
            0.0
        )

    # Calculate urea to calcium ratio if both exist
    if 'urea' in df_engineered.columns and 'calc' in df_engineered.columns:
        df_engineered['urea_calc_ratio'] = np.where(
            df_engineered['calc'] != 0,
            df_engineered['urea'] / df_engineered['calc'],
            0.0
        )

    logger.info(f"Engineered dataset features. New shape: {df_engineered.shape}")
    return df_engineered


def extract_metadata(
    df: pd.DataFrame,
    target_col: str,
    num_cols: List[str],
    cat_cols: List[str],
    train_len: int,
    val_len: int,
    test_len: int
) -> Dict[str, Any]:
    """Generates structured metadata dictionary for ML preprocessing."""
    metadata = {
        "dataset_name": "StoneSense-AI Urine Analysis / Kidney Dataset",
        "target_column": target_col,
        "feature_names": [c for c in df.columns if c != target_col],
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "preprocessing_steps": [
            "duplicate_row_removal",
            "missing_target_row_dropping",
            "median_numerical_imputation",
            "mode_categorical_imputation",
            "standard_scaler_numerical_normalization",
            "one_hot_categorical_encoding",
            "stratified_70_15_15_dataset_split"
        ],
        "dataset_splits": {
            "train_size": train_len,
            "validation_size": val_len,
            "test_size": test_len,
            "total_size": train_len + val_len + test_len
        }
    }
    return metadata
