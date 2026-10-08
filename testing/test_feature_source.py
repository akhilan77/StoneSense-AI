"""Unit tests validating authoritative feature source and ordering alignment."""

import sys
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.artifacts import get_feature_columns
from ml.risk_models.registry import registry
from backend.app.utils.preprocessing_utils import prepare_tabular_inputs, get_expected_feature_columns



def test_feature_source_authoritative_artifact():
    """Verifies that feature columns match the authoritative feature_columns.pkl artifact."""
    artifact_path = PROJECT_ROOT / "ml" / "artifacts" / "feature_columns.pkl"
    assert artifact_path.exists()
    disk_features = joblib.load(artifact_path)
    assert disk_features == ["gravity", "ph", "osmo", "cond", "urea", "calc"]
    assert get_feature_columns() == disk_features


def test_backend_feature_source_alignment():
    """Verifies backend preprocessing utility loads authoritative features in exact order."""
    backend_cols = get_expected_feature_columns()
    authoritative_cols = get_feature_columns()
    assert backend_cols == authoritative_cols


def test_model_registry_feature_alignment():
    """Verifies active model in registry uses identical feature list and ordering."""
    model = registry.get_active_model()
    authoritative_cols = get_feature_columns()
    assert model.feature_names == authoritative_cols


def test_prepare_tabular_inputs_ordering():
    """Verifies prepare_tabular_inputs returns DataFrame with exact authoritative columns and ordering."""
    raw_dict = {
        "calc": 5.2,
        "urea": 300,
        "urine_ph": 6.5,
        "urine_specific_gravity": 1.020,
        "osmolality": 600,
        "conductivity": 25.0
    }
    df = prepare_tabular_inputs(raw_dict)
    assert list(df.columns) == ["gravity", "ph", "osmo", "cond", "urea", "calc"]
    assert df["gravity"].iloc[0] == 1.020
    assert df["ph"].iloc[0] == 6.5
    assert df["calc"].iloc[0] == 5.2
