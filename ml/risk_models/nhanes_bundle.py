"""Self-contained inference bundle for the NHANES v003 tabular risk models.

A bundle packs everything needed to score one raw clinical-profile row:
fitted preprocessor -> fitted gradient-boosting / EBM estimator -> frozen probability calibrator.
It also produces local per-feature contributions (SHAP TreeExplainer for tree models,
native additive term scores for the EBM) aggregated back to the raw input features.

This module is production code: it is imported by the training helpers, the registry wrappers and
the backend. It must never import notebooks.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Columns that must never be a model input (identifiers, the label and its proxies, survey design).
BANNED_COLUMNS = (
    "SEQN", "CYCLE", "target_stone", "KIQ026", "KIQ029",
    "WTMEC2YR", "WTINT2YR", "SDMVPSU", "SDMVSTRA",
)

TREE_FAMILIES = ("xgboost", "lightgbm", "catboost")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ProbCalibrator:
    """Maps raw model probabilities to calibrated probabilities (isotonic or sigmoid/Platt)."""

    def __init__(self, method: str, model: Any = None):
        if method not in ("isotonic", "sigmoid", "identity"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.model = model

    @classmethod
    def fit(cls, p_raw: np.ndarray, y: np.ndarray, method: str) -> "ProbCalibrator":
        p_raw = np.asarray(p_raw, dtype=float)
        y = np.asarray(y).astype(int)
        if method == "isotonic":
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw, y)
        elif method == "sigmoid":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_raw).reshape(-1, 1), y)
        else:
            m = None
        return cls(method, m)

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype=float)
        if self.method == "isotonic":
            return np.asarray(self.model.predict(p_raw), dtype=float)
        if self.method == "sigmoid":
            return self.model.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
        return p_raw


class NhanesBundle:
    """Preprocessor + estimator + calibrator for one (model family, feature variant)."""

    def __init__(
        self,
        model_name: str,
        family: str,
        variant: str,
        input_features: Sequence[str],
        categorical_features: Sequence[str],
        preprocessor: Any,
        estimator: Any,
        calibrator: ProbCalibrator,
        thresholds: Dict[str, Optional[float]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        banned = sorted(set(input_features) & set(BANNED_COLUMNS))
        if banned:
            raise ValueError(f"Banned columns used as model inputs: {banned}")
        self.model_name = model_name
        self.family = family
        self.variant = variant
        self.input_features: List[str] = list(input_features)
        self.categorical_features: List[str] = list(categorical_features)
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.calibrator = calibrator
        self.thresholds = dict(thresholds)
        self.metadata = dict(metadata or {})
        self._explainer: Any = None

    # ------------------------------------------------------------------ plumbing
    @property
    def threshold(self) -> float:
        """Default decision threshold: the accuracy-focused operating point of nhanes_v003b bundles; the
        balanced-accuracy threshold of the legacy nhanes_v003 bundles (which have no 'accuracy' key)."""
        return self.operating_threshold(None)

    def operating_threshold(self, point: Optional[str] = None) -> float:
        """Threshold of a named operating point ('accuracy' | 'balanced'); None = the default described above."""
        order = [point] if point else ["accuracy", "balanced", "balanced_accuracy"]
        for key in order:
            if self.thresholds.get(key) is not None:
                return float(self.thresholds[key])
        raise KeyError(f"No threshold stored for operating point {point!r}: {sorted(self.thresholds)}")

    @property
    def numeric_features(self) -> List[str]:
        return [f for f in self.input_features if f not in self.categorical_features]

    def transformed_feature_names(self) -> List[str]:
        names = list(self.numeric_features)
        if self.categorical_features:
            enc = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            names += enc.get_feature_names_out(self.categorical_features).tolist()
        return names

    def _raw_feature_for_column(self) -> List[str]:
        """Raw input feature behind every transformed column."""
        out = list(self.numeric_features)
        for col in self.categorical_features:
            enc = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cats = enc.categories_[self.categorical_features.index(col)]
            out += [col] * len(cats)
        return out

    def frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            df = pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=float)), columns=self.input_features)
        missing = [c for c in self.input_features if c not in df.columns]
        if missing:
            raise KeyError(f"Missing model input columns: {missing}")
        return df[self.input_features].astype(float)

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(self.frame(X)), dtype=np.float32)

    # ------------------------------------------------------------------ prediction
    def predict_proba_raw(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self.transform(X)))[:, 1]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Calibrated probability of a reported kidney stone, shape (n,)."""
        return self.calibrator.predict(self.predict_proba_raw(X))

    # ------------------------------------------------------------------ explanations
    def contributions(self, X: Any) -> np.ndarray:
        """Per-raw-feature additive contributions (log-odds scale, uncalibrated), shape (n, n_features)."""
        Xt = self.transform(X)
        if self.family in TREE_FAMILIES:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.estimator)
            sv = self._explainer.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) == 2 else sv[0]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            col_scores = sv
            col_to_raw = self._raw_feature_for_column()
            out = np.zeros((len(Xt), len(self.input_features)))
            for j, raw in enumerate(col_to_raw):
                out[:, self.input_features.index(raw)] += col_scores[:, j]
            return out
        if self.family == "ebm":
            term_scores = np.asarray(self.estimator.eval_terms(Xt))
            col_to_raw = self._raw_feature_for_column()
            out = np.zeros((len(Xt), len(self.input_features)))
            for t, feats in enumerate(self.estimator.term_features_):
                raws = sorted({col_to_raw[i] for i in feats})
                for raw in raws:  # interaction scores are split evenly across their member features
                    out[:, self.input_features.index(raw)] += term_scores[:, t] / len(raws)
            return out
        if self.family == "logreg":  # standardised inputs: contribution = coefficient * standardised value
            col_scores = Xt * np.asarray(self.estimator.coef_)[0]
            col_to_raw = self._raw_feature_for_column()
            out = np.zeros((len(Xt), len(self.input_features)))
            for j, raw in enumerate(col_to_raw):
                out[:, self.input_features.index(raw)] += col_scores[:, j]
            return out
        raise ValueError(f"No explanation method for family '{self.family}'")

    # ------------------------------------------------------------------ pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_explainer"] = None
        return state
