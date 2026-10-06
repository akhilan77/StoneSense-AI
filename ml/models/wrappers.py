"""Model Adapters for Tabular Kidney Stone Risk Models.

Implements BaseRiskModel wrappers for:
- LogisticRegressionRiskModel (model_family: logistic_regression)
- RandomForestRiskModel (model_family: random_forest)
- XGBoostRiskModel (model_family: xgboost)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib

try:
    import shap
except ImportError:
    shap = None

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

from .base import BaseRiskModel


def format_shap_explanation(
    feature_names: List[str],
    raw_values: np.ndarray,
    summary_text: Optional[str] = None
) -> Dict[str, Any]:
    """Helper to convert raw SHAP values into the standardized StoneSense schema."""
    # Ensure 1D array
    vals = np.asarray(raw_values).flatten()
    contributions = {feat: float(val) for feat, val in zip(feature_names, vals)}
    sorted_features = sorted(contributions.keys(), key=lambda k: abs(contributions[k]), reverse=True)

    directions = {}
    for feat, val in contributions.items():
        if val > 1e-6:
            directions[feat] = "increases"
        elif val < -1e-6:
            directions[feat] = "decreases"
        else:
            directions[feat] = "neutral"

    default_summary = (
        "The strongest SHAP contributors influenced the model output for this request; "
        "they do not establish causation."
    )

    return {
        "top_features": sorted_features,
        "feature_contributions": contributions,
        "feature_directions": directions,
        "summary": summary_text or default_summary,
    }


class LogisticRegressionRiskModel(BaseRiskModel):
    """Logistic Regression Risk Model adapter using LinearExplainer for explainability."""

    def __init__(
        self,
        version_tag: str = "logistic_regression_v001",
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.50,
        estimator: Optional[LogisticRegression] = None,
        background_data: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_family="logistic_regression",
            version_tag=version_tag,
            feature_names=feature_names,
            threshold=threshold,
            metadata=metadata,
        )
        self.estimator = estimator or LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        self.background_data = background_data
        self._explainer: Any = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "LogisticRegressionRiskModel":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.estimator.fit(X_arr, y_arr)
        self.background_data = X_arr
        self._explainer = None
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X)
        return self.estimator.predict_proba(X_arr)

    def _get_explainer(self) -> Any:
        if self._explainer is None and shap is not None:
            if self.background_data is not None:
                bg = self.background_data
            else:
                # Default background in standardized space is zero vector
                bg = np.zeros((1, len(self.feature_names)))
            self._explainer = shap.LinearExplainer(self.estimator, bg)
        return self._explainer

    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Calculates linear SHAP feature contributions for the input sample."""
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        explainer = self._get_explainer()
        if explainer is not None:
            df_trans = pd.DataFrame(X_arr, columns=self.feature_names)
            explanation = explainer(df_trans)
            vals = explanation.values[0]
            # If multi-class output format (e.g. 2 classes in output dimension), pick positive class index 1
            if isinstance(vals, np.ndarray) and vals.ndim > 1:
                vals = vals[:, 1] if vals.shape[1] == 2 else vals[:, 0]
        else:
            # Analytical linear fallback if shap library is unavailable:
            # phi_i = coef_i * (x_i - mean_i)
            coefs = self.estimator.coef_[0]
            bg_mean = np.mean(self.background_data, axis=0) if self.background_data is not None else np.zeros_like(coefs)
            vals = coefs * (X_arr[0] - bg_mean)

        return format_shap_explanation(self.feature_names, vals)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Clear cached explainer before pickling to ensure portability
        self._explainer = None
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticRegressionRiskModel":
        obj = joblib.load(path)
        if isinstance(obj, LogisticRegression):
            return cls(version_tag=Path(path).stem, estimator=obj)
        elif isinstance(obj, cls) or type(obj).__name__ == cls.__name__:
            # Re-wrap if loaded under alternative module namespace
            instance = cls(
                version_tag=getattr(obj, "version_tag", Path(path).stem),
                feature_names=getattr(obj, "feature_names", None),
                threshold=getattr(obj, "threshold", 0.50),
                estimator=getattr(obj, "estimator", None),
                background_data=getattr(obj, "background_data", None),
                metadata=getattr(obj, "metadata", None),
            )
            return instance
        elif hasattr(obj, "predict_proba"):
            return cls(version_tag=Path(path).stem, estimator=obj)
        else:
            raise TypeError(f"Loaded object from {path} is of unexpected type: {type(obj)}")


class RandomForestRiskModel(BaseRiskModel):
    """Random Forest Risk Model adapter using TreeExplainer for explainability."""

    def __init__(
        self,
        version_tag: str = "random_forest_v001",
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.50,
        estimator: Optional[RandomForestClassifier] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_family="random_forest",
            version_tag=version_tag,
            feature_names=feature_names,
            threshold=threshold,
            metadata=metadata,
        )
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
        if hasattr(self.estimator, "estimator"):
            self.estimator = self.estimator.estimator
        self._explainer: Any = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "RandomForestRiskModel":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.estimator.fit(X_arr, y_arr)
        self._explainer = None
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X)
        return self.estimator.predict_proba(X_arr)

    def _get_explainer(self) -> Any:
        if self._explainer is None and shap is not None:
            underlying = getattr(self.estimator, "estimator", self.estimator)
            self._explainer = shap.TreeExplainer(underlying)
        return self._explainer

    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        explainer = self._get_explainer()
        if explainer is not None:
            df_trans = pd.DataFrame(X_arr, columns=self.feature_names)
            explanation = explainer(df_trans)
            vals = explanation.values[0]
            if isinstance(vals, np.ndarray) and vals.ndim > 1:
                vals = vals[:, 1] if vals.shape[1] == 2 else vals[:, 0]
        else:
            importances = getattr(self.estimator, "feature_importances_", np.ones(len(self.feature_names)))
            vals = importances * X_arr[0]

        return format_shap_explanation(self.feature_names, vals)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._explainer = None
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RandomForestRiskModel":
        obj = joblib.load(path)
        if isinstance(obj, RandomForestClassifier):
            return cls(version_tag=Path(path).stem, estimator=obj)
        elif isinstance(obj, cls) or type(obj).__name__ == cls.__name__:
            est = getattr(obj, "estimator", None)
            if hasattr(est, "estimator"):
                est = est.estimator
            return cls(
                version_tag=getattr(obj, "version_tag", Path(path).stem),
                feature_names=getattr(obj, "feature_names", None),
                threshold=getattr(obj, "threshold", 0.50),
                estimator=est,
                metadata=getattr(obj, "metadata", None),
            )
        elif hasattr(obj, "predict_proba"):
            return cls(version_tag=Path(path).stem, estimator=obj)
        else:
            raise TypeError(f"Loaded object from {path} is of unexpected type: {type(obj)}")


class XGBoostRiskModel(BaseRiskModel):
    """XGBoost Risk Model adapter using TreeExplainer for explainability."""

    def __init__(
        self,
        version_tag: str = "xgboost_v001",
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.50,
        estimator: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_family="xgboost",
            version_tag=version_tag,
            feature_names=feature_names,
            threshold=threshold,
            metadata=metadata,
        )
        self.estimator = estimator or XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )
        if hasattr(self.estimator, "estimator"):
            self.estimator = self.estimator.estimator
        self._explainer: Any = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "XGBoostRiskModel":
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.estimator.fit(X_arr, y_arr)
        self._explainer = None
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X)
        return self.estimator.predict_proba(X_arr)

    def _get_explainer(self) -> Any:
        if self._explainer is None and shap is not None:
            underlying = getattr(self.estimator, "estimator", self.estimator)
            self._explainer = shap.TreeExplainer(underlying)
        return self._explainer

    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        explainer = self._get_explainer()
        if explainer is not None:
            df_trans = pd.DataFrame(X_arr, columns=self.feature_names)
            explanation = explainer(df_trans)
            vals = explanation.values[0]
            if isinstance(vals, np.ndarray) and vals.ndim > 1:
                vals = vals[:, 1] if vals.shape[1] == 2 else vals[:, 0]
        else:
            importances = getattr(self.estimator, "feature_importances_", np.ones(len(self.feature_names)))
            vals = importances * X_arr[0]

        return format_shap_explanation(self.feature_names, vals)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._explainer = None
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBoostRiskModel":
        obj = joblib.load(path)
        if isinstance(obj, (XGBClassifier, getattr(joblib, "XGBClassifier", type(None)))):
            return cls(version_tag=Path(path).stem, estimator=obj)
        elif isinstance(obj, cls) or type(obj).__name__ == cls.__name__:
            est = getattr(obj, "estimator", None)
            if hasattr(est, "estimator"):
                est = est.estimator
            return cls(
                version_tag=getattr(obj, "version_tag", Path(path).stem),
                feature_names=getattr(obj, "feature_names", None),
                threshold=getattr(obj, "threshold", 0.50),
                estimator=est,
                metadata=getattr(obj, "metadata", None),
            )
        elif hasattr(obj, "predict_proba"):
            return cls(version_tag=Path(path).stem, estimator=obj)
        else:
            raise TypeError(f"Loaded object from {path} is of unexpected type: {type(obj)}")



FAMILY_MODEL_MAP = {
    "logistic_regression": LogisticRegressionRiskModel,
    "random_forest": RandomForestRiskModel,
    "xgboost": XGBoostRiskModel,
}
