"""Model Abstraction and Registry Package for StoneSense-AI."""

from .base import BaseRiskModel
from .wrappers import (
    LogisticRegressionRiskModel,
    RandomForestRiskModel,
    XGBoostRiskModel,
    XGBoostNhanesRiskModel,
    CatBoostRiskModel,
    LightGBMRiskModel,
    EBMRiskModel,
    NhanesBundleRiskModel,
    FAMILY_MODEL_MAP,
)
from .nhanes_bundle import NhanesBundle
from .registry import ModelRegistry, registry

__all__ = [
    "BaseRiskModel",
    "LogisticRegressionRiskModel",
    "RandomForestRiskModel",
    "XGBoostRiskModel",
    "XGBoostNhanesRiskModel",
    "CatBoostRiskModel",
    "LightGBMRiskModel",
    "EBMRiskModel",
    "NhanesBundleRiskModel",
    "NhanesBundle",
    "FAMILY_MODEL_MAP",
    "ModelRegistry",
    "registry",
]
