"""Tests validating prediction parity and threshold configuration."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.risk_models.registry import registry
from app.services.prediction_service import PredictionService
from app.services.model_loader import model_loader

DATASET_PATH = PROJECT_ROOT / "ml" / "datasets" / "kidneyData.csv"


def test_prediction_numerical_parity():
    """Verifies that the registry LogisticRegression model matches candidate_risk_model.pkl within 1e-7 tolerance."""
    raw_model_path = PROJECT_ROOT / "ml" / "models" / "candidate_risk_model.pkl"
    pipe_path = PROJECT_ROOT / "ml" / "artifacts" / "preprocessing_pipeline.pkl"
    dataset_path = DATASET_PATH

    raw_model = joblib.load(raw_model_path)
    pipe = joblib.load(pipe_path)
    reg_model = registry.get_model("logistic_regression_v001")

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        for c in ["Unnamed: 0", "target", "Class", "label"]:
            if c in df.columns:
                df = df.drop(columns=[c])
        X = df[reg_model.feature_names]
    else:
        # Synthetic clinical fixture for CI environments where raw datasets are gitignored
        X = pd.DataFrame({
            "gravity": [1.010, 1.015, 1.020, 1.025, 1.030, 1.012, 1.018],
            "ph": [5.5, 6.0, 6.5, 7.0, 7.5, 5.8, 6.8],
            "osmo": [400, 500, 600, 700, 800, 450, 750],
            "cond": [15.0, 20.0, 25.0, 30.0, 35.0, 18.0, 32.0],
            "urea": [150, 250, 350, 450, 550, 200, 500],
            "calc": [2.0, 4.0, 6.0, 8.0, 10.0, 3.0, 7.5],
        })

    X_trans = pipe.transform(X)

    raw_probs = raw_model.predict_proba(X_trans)
    reg_probs = reg_model.predict_proba(X_trans)

    max_diff = np.max(np.abs(raw_probs - reg_probs))
    assert max_diff < 1e-7, f"Max probability difference {max_diff} exceeded tolerance 1e-7"

    raw_preds = raw_model.predict(X_trans)
    reg_preds = reg_model.predict(X_trans)
    np.testing.assert_array_equal(raw_preds, reg_preds)


def test_threshold_decision_logic():
    """Verifies threshold boundary logic: prob >= threshold -> High, prob < threshold -> Low."""
    model = registry.get_model("logistic_regression_v001")
    model.threshold = 0.50

    # Mock predictions
    mock_X = np.zeros((3, 6))

    # Test custom threshold logic on wrapper
    model.threshold = 0.40
    # Create wrapper with deterministic proba
    class MockModel(model.__class__):
        def predict_proba(self, X):
            return np.array([
                [0.7, 0.3],   # 0.30 < 0.40 -> 0 (Low)
                [0.6, 0.4],   # 0.40 >= 0.40 -> 1 (High)
                [0.1, 0.9],   # 0.90 >= 0.40 -> 1 (High)
            ])

    mock = MockModel(threshold=0.40)
    preds = mock.predict(mock_X)
    assert list(preds) == [0, 1, 1]


def test_prediction_service_returns_expected_structure():
    """Verifies that PredictionService.predict_risk executes successfully on clinical dictionary."""
    model_loader.load_ml_model()
    patient_data = {
        "gravity": 1.015,
        "ph": 6.2,
        "osmo": 550,
        "cond": 22.0,
        "urea": 250,
        "calc": 4.5
    }
    res = PredictionService.predict_risk(patient_data)
    assert "probability" in res
    assert "risk" in res
    assert res["risk"] in ["Low", "High"]
    assert 0.0 <= res["probability"] <= 1.0
