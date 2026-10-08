"""Backward-compatible re-exports from ml.risk_models.wrappers for pickle deserialization."""
from ml.risk_models.wrappers import (
    BaseRiskModel,
    LogisticRegressionRiskModel,
    RandomForestRiskModel,
    XGBoostRiskModel,
    FAMILY_MODEL_MAP,
    format_shap_explanation,
)

__all__ = [
    "BaseRiskModel",
    "LogisticRegressionRiskModel",
    "RandomForestRiskModel",
    "XGBoostRiskModel",
    "FAMILY_MODEL_MAP",
    "format_shap_explanation",
]
