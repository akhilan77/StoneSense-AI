"""Unit and integration tests for ModelRegistry and BaseRiskModel adapters."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.models.base import BaseRiskModel
from ml.models.wrappers import (
    LogisticRegressionRiskModel,
    RandomForestRiskModel,
    XGBoostRiskModel,
)
from ml.models.registry import ModelRegistry, registry



def test_registry_active_model_loading():
    """Verifies that the default registry loads the active model (logistic_regression_v001)."""
    model = registry.get_active_model()
    assert isinstance(model, BaseRiskModel)
    assert model.model_family == "logistic_regression"
    assert model.version_tag == "logistic_regression_v001"
    assert model.threshold == 0.50
    assert len(model.feature_names) == 6


def test_registry_load_by_family():
    """Verifies resolving models by model family."""
    lr = registry.get_model_by_family("logistic_regression")
    assert isinstance(lr, LogisticRegressionRiskModel)
    assert lr.model_family == "logistic_regression"

    xgb = registry.get_model_by_family("xgboost")
    assert isinstance(xgb, XGBoostRiskModel)
    assert xgb.model_family == "xgboost"


def test_registry_load_by_version():
    """Verifies loading specific model versions."""
    lr = registry.get_model("logistic_regression_v001")
    assert lr.version_tag == "logistic_regression_v001"

    xgb = registry.get_model("xgboost_v001")
    assert xgb.version_tag == "xgboost_v001"


def test_registry_invalid_family_error():
    """Verifies exception when querying unsupported model family."""
    with pytest.raises(ValueError, match="Invalid model family"):
        registry.get_model_by_family("unsupported_neural_net")


def test_registry_missing_version_error():
    """Verifies exception when requesting non-existent version."""
    with pytest.raises(KeyError, match="not found in registry"):
        registry.get_model("non_existent_v999")


def test_register_and_switch_active_model(tmp_path):
    """Verifies registering a new model version and switching active pointers in an isolated registry."""
    test_manifest = tmp_path / "manifest.json"
    test_versions = tmp_path / "versions"
    reg = ModelRegistry(versions_dir=test_versions, manifest_path=test_manifest)

    # 1. Register Random Forest model
    rf = RandomForestRiskModel(version_tag="random_forest_v001")
    X_mock = np.random.randn(20, 6)
    y_mock = np.random.randint(0, 2, size=20)
    rf.fit(X_mock, y_mock)

    v_tag = reg.register_model(rf, metadata={"metrics": {"accuracy": 0.85}}, set_active=True)
    assert v_tag == "random_forest_v001"
    assert reg.get_active_model().version_tag == "random_forest_v001"

    # 2. Register Logistic Regression model
    lr = LogisticRegressionRiskModel(version_tag="lr_test_v002")
    lr.fit(X_mock, y_mock)
    reg.register_model(lr, set_active=False)

    # Active should still be RF
    assert reg.get_active_model().version_tag == "random_forest_v001"

    # Switch active to LR
    reg.set_active_model("lr_test_v002")
    assert reg.get_active_model().version_tag == "lr_test_v002"
    assert reg.get_active_model().model_family == "logistic_regression"
