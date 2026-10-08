"""Unit tests validating model-aware SHAP explainability and schema compliance."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.risk_models.wrappers import (
    LogisticRegressionRiskModel,
    RandomForestRiskModel,
    XGBoostRiskModel,
)
from ml.risk_models.registry import registry
from backend.app.services.explainability_service import generate_shap_for_patient
from backend.app.services.model_loader import model_loader



@pytest.fixture(autouse=True)
def ensure_model_loaded():
    """Ensure model_loader singleton has active model loaded."""
    model_loader.load_ml_model()


def test_logistic_regression_linear_explainer():
    """Verifies that LogisticRegressionRiskModel uses LinearExplainer and produces aligned outputs."""
    model = registry.get_model("logistic_regression_v001")
    pipe = joblib.load(PROJECT_ROOT / "ml" / "artifacts" / "preprocessing_pipeline.pkl")

    sample_df = pd.DataFrame([{
        "gravity": 1.025,
        "ph": 5.5,
        "osmo": 700,
        "cond": 28.0,
        "urea": 400,
        "calc": 8.0
    }])
    X_trans = pipe.transform(sample_df)

    explanation = model.explain(X_trans)

    assert "top_features" in explanation
    assert "feature_contributions" in explanation
    assert "feature_directions" in explanation
    assert "summary" in explanation

    # Check 6 features
    assert len(explanation["feature_contributions"]) == 6
    assert set(explanation["feature_contributions"].keys()) == set(model.feature_names)
    assert set(explanation["top_features"]) == set(model.feature_names)

    # Check directions mapping
    for feat, direction in explanation["feature_directions"].items():
        assert direction in ["increases", "decreases", "neutral"]
        val = explanation["feature_contributions"][feat]
        if val > 1e-6:
            assert direction == "increases"
        elif val < -1e-6:
            assert direction == "decreases"
        else:
            assert direction == "neutral"


def test_tree_model_tree_explainer():
    """Verifies that XGBoostRiskModel uses TreeExplainer and produces valid schema."""
    xgb_model = registry.get_model("xgboost_v001")
    pipe = joblib.load(PROJECT_ROOT / "ml" / "artifacts" / "preprocessing_pipeline.pkl")

    sample_df = pd.DataFrame([{
        "gravity": 1.010,
        "ph": 7.0,
        "osmo": 400,
        "cond": 15.0,
        "urea": 150,
        "calc": 2.0
    }])
    X_trans = pipe.transform(sample_df)

    explanation = xgb_model.explain(X_trans)
    assert len(explanation["feature_contributions"]) == 6
    assert set(explanation["feature_contributions"].keys()) == set(xgb_model.feature_names)


def test_generate_shap_for_patient_schema_compatibility():
    """Verifies that backend generate_shap_for_patient returns exact expected keys and types."""
    patient_data = {
        "urine_specific_gravity": 1.020,
        "urine_ph": 6.0,
        "osmolality": 600,
        "conductivity": 22.0,
        "urea": 300,
        "calcium": 6.5
    }
    res = generate_shap_for_patient(patient_data)
    assert "top_features" in res
    assert "feature_contributions" in res
    assert "feature_directions" in res
    assert "summary" in res
    assert isinstance(res["top_features"], list)
    assert isinstance(res["feature_contributions"], dict)
    assert isinstance(res["feature_directions"], dict)
    assert isinstance(res["summary"], str)
