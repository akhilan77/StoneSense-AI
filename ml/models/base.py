"""Base Risk Model Abstraction for Tabular Kidney Stone Risk Models.

Defines the common interface (fit, predict, predict_proba, explain, save, load)
for all tabular clinical risk models in StoneSense-AI.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd


class BaseRiskModel(ABC):
    """Abstract Base Class for tabular clinical risk prediction models."""

    def __init__(
        self,
        model_family: str,
        version_tag: str,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.50,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_family = model_family
        self.version_tag = version_tag
        self.feature_names = feature_names or ["gravity", "ph", "osmo", "cond", "urea", "calc"]
        self.threshold = float(threshold)
        self.metadata = metadata or {}
        self.estimator: Any = None
        self.background_data: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "BaseRiskModel":
        """Fits the underlying model on preprocessed training data."""
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns predicted class probabilities of shape (N, 2)."""
        pass

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts binary class labels (0 or 1) applying the configured decision threshold."""
        probs = self.predict_proba(X)
        positive_probs = probs[:, 1] if probs.ndim == 2 else probs
        return (positive_probs >= self.threshold).astype(int)

    @abstractmethod
    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Generates structured local SHAP explanations for the input observations."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Serializes the wrapped model instance and its parameters to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BaseRiskModel":
        """Loads and deserializes a wrapped model instance from disk."""
        pass
