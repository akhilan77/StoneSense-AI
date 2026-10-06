"""Authoritative ML Artifacts Loader and Single Source of Truth."""

from pathlib import Path
from typing import List, Any
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
PREPROCESSING_PIPELINE_PATH = ARTIFACTS_DIR / "preprocessing_pipeline.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

_CACHED_FEATURES: List[str] = []


def get_feature_columns() -> List[str]:
    """Returns the authoritative list of tabular feature column names."""
    global _CACHED_FEATURES
    if not _CACHED_FEATURES:
        if FEATURE_COLUMNS_PATH.exists():
            _CACHED_FEATURES = list(joblib.load(FEATURE_COLUMNS_PATH))
        else:
            _CACHED_FEATURES = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
    return list(_CACHED_FEATURES)


def get_preprocessing_pipeline() -> Any:
    """Returns the authoritative fitted ColumnTransformer preprocessing pipeline."""
    if PREPROCESSING_PIPELINE_PATH.exists():
        return joblib.load(PREPROCESSING_PIPELINE_PATH)
    raise FileNotFoundError(f"Authoritative preprocessing pipeline not found at {PREPROCESSING_PIPELINE_PATH}")
