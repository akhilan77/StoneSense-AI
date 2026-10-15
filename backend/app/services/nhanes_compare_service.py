"""Four-model NHANES clinical-profile comparison (EBM, CatBoost, XGBoost, LightGBM).

Validates the submitted form against ml/artifacts/nhanes_v003/feature_spec.json, scores the four registered
``*_nhanes_v003`` models through the model registry, and attaches each model's stored held-out TEST metrics.
Nothing is trained or evaluated here, and the submitted values are never persisted.
"""
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("NhanesCompareService")

SPEC_PATH = PROJECT_ROOT / "ml" / "artifacts" / "nhanes_v003" / "feature_spec.json"
MODEL_TAGS: Tuple[str, ...] = (
    "ebm_nhanes_v003b", "catboost_nhanes_v003b", "xgboost_nhanes_v003b", "lightgbm_nhanes_v003b",
)
OPERATING_POINTS: Tuple[str, ...] = ("accuracy", "balanced")
DEFAULT_OPERATING_POINT = "accuracy"
OP_LABEL = {"accuracy": "Accuracy-focused (>70% accuracy)", "balanced": "Balanced (max balanced accuracy)"}
OP_RULE = {
    "accuracy": "maximum sensitivity subject to validation accuracy >= 72% (validation cycle 2015-2016)",
    "balanced": "maximum balanced accuracy on the validation cycle 2015-2016",
}
DISPLAY_NAME = {"ebm": "EBM", "catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
METRICS_NOTE = (
    "Metrics describe each model's performance on held-out 2017-2018 NHANES test data. "
    "They are not the accuracy of the prediction for this patient."
)
DISCLAIMER = (
    "Research prototype. Models are trained on US NHANES survey data with a self-reported kidney-stone label, "
    "are pending review and are not approved for clinical use. They estimate statistical association, not diagnosis."
)
TOP_N_FACTORS = 5


class FeatureValidationError(ValueError):
    """Raised with a list of human-readable problems when the submitted features are invalid."""

    def __init__(self, problems: List[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class ModelsUnavailableError(RuntimeError):
    """Raised when one or more of the four registered models cannot be loaded."""


class NhanesCompareService:
    def __init__(self, registry: Any = None, spec_path: Optional[Path] = None, model_tags: Tuple[str, ...] = MODEL_TAGS):
        self._registry = registry
        self._spec_path = Path(spec_path or SPEC_PATH)
        self._model_tags = tuple(model_tags)
        self._spec: Optional[Dict[str, Any]] = None
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ spec / validation
    @property
    def registry(self) -> Any:
        if self._registry is None:
            from ml.risk_models import registry as default_registry

            self._registry = default_registry
        return self._registry

    def get_spec(self) -> Dict[str, Any]:
        if self._spec is None:
            if not self._spec_path.exists():
                raise FileNotFoundError(f"Feature specification not found at {self._spec_path}")
            with open(self._spec_path, "r", encoding="utf-8") as f:
                self._spec = json.load(f)
        return self._spec

    def validate(self, features: Dict[str, Optional[float]]) -> Tuple[Dict[str, float], List[str]]:
        """Returns (full feature row incl. derived values, warnings) or raises FeatureValidationError."""
        spec = self.get_spec()
        by_name = {f["name"]: f for f in spec["features"]}
        problems: List[str] = []
        warnings: List[str] = []

        unknown = sorted(set(features) - set(by_name))
        if unknown:
            problems.append(f"Unknown feature(s): {', '.join(unknown)}")
        for name in set(features) & set(by_name):
            if by_name[name]["type"] == "derived":
                problems.append(f"{name} is derived by the server and must not be submitted")

        row: Dict[str, float] = {}
        for name, meta in by_name.items():
            if meta["type"] == "derived":
                continue
            raw = features.get(name)
            label = meta.get("label", name)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                if meta.get("required", True):
                    problems.append(f"{label} ({name}) is required")
                else:
                    row[name] = float("nan")
                continue
            value = float(raw)
            if not math.isfinite(value):
                problems.append(f"{label} ({name}) must be a finite number")
                continue
            options = meta.get("options")
            if options:
                allowed = [float(o["value"]) for o in options]
                if not any(abs(value - a) < 1e-9 for a in allowed):
                    problems.append(f"{label} ({name}) must be one of {allowed}")
                    continue
            elif not (float(meta["min"]) <= value <= float(meta["max"])):
                problems.append(f"{label} ({name}) must be between {meta['min']} and {meta['max']} {meta.get('unit', '')}".strip())
                continue
            row[name] = value
        if problems:
            raise FeatureValidationError(problems)

        for name, meta in by_name.items():  # derived values
            if meta["type"] != "derived":
                continue
            num, den = (row.get(src, float("nan")) for src in meta["derived_from"])
            value = num / den * 100.0 if den and not math.isnan(num) and not math.isnan(den) else float("nan")
            if not math.isnan(value) and not (meta["min"] <= value <= meta["max"]):
                clipped = min(max(value, float(meta["min"])), float(meta["max"]))
                warnings.append(f"{meta['label']} computed as {value:.3g} {meta['unit']}; clipped to {clipped:g} (model training range).")
                value = clipped
            row[name] = value
        return row, warnings

    # ------------------------------------------------------------------ models
    def _load_models(self) -> Dict[str, Any]:
        with self._lock:
            missing, reasons = [], []
            for tag in self._model_tags:
                if tag in self._models:
                    continue
                try:
                    self._models[tag] = self.registry.get_model(tag)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Cannot load %s: %s", tag, exc)
                    missing.append(tag)
                    reasons.append(f"{tag}: {type(exc).__name__}: {exc}")
            if missing:
                raise ModelsUnavailableError(
                    f"Registered NHANES v003 model(s) unavailable: {', '.join(missing)}. "
                    "Reasons: " + " | ".join(reasons) + ". If a Python module is missing, run: "
                    "pip install -r requirements.txt (catboost, lightgbm and interpret are required); otherwise run "
                    "ml/training/run_nhanes_notebooks.py to train and register the models."
                )
            return dict(self._models)

    def _entry(self, tag: str) -> Dict[str, Any]:
        for entry in self.registry.list_models():
            if entry.get("version_tag") == tag:
                return dict(entry)
        raise KeyError(tag)

    @staticmethod
    def _operating_points(entry: Dict[str, Any], prob: float) -> Dict[str, Dict[str, Any]]:
        """Both operating points of a registry entry with this patient's risk level at each. Falls back to the legacy
        nhanes_v003 layout (a single balanced-accuracy threshold) when the entry has no ``operating_points``."""
        stored = entry.get("operating_points")
        tm = entry.get("test_metrics") or {}
        out: Dict[str, Dict[str, Any]] = {}
        for pt in OPERATING_POINTS:
            if stored and pt in stored:
                src = stored[pt]
                thr, test, val = src.get("threshold"), src.get("test"), src.get("validation")
            else:  # legacy entry
                legacy = tm.get("at_balanced_accuracy_threshold") if pt == "balanced" else tm.get("at_accuracy_ge_75_threshold")
                thr, test, val = (legacy or {}).get("threshold"), legacy, None
            out[pt] = {
                "label": OP_LABEL[pt], "rule": OP_RULE[pt],
                "threshold": None if thr is None else round(float(thr), 4),
                "risk_level": None if thr is None else ("High" if prob >= float(thr) else "Low"),
                "validation": val, "test": test,
            }
        return out

    def compare(self, features: Dict[str, Optional[float]], operating_point: str = DEFAULT_OPERATING_POINT) -> Dict[str, Any]:
        if operating_point not in OPERATING_POINTS:
            raise FeatureValidationError([f"operating_point must be one of {list(OPERATING_POINTS)}"])
        start = time.perf_counter()
        row, warnings = self.validate(features)
        models = self._load_models()
        spec_by_name = {f["name"]: f for f in self.get_spec()["features"]}
        blank = {name for name, value in row.items() if isinstance(value, float) and math.isnan(value)}
        if blank:
            labels = ", ".join(spec_by_name.get(n, {}).get("label", n) for n in sorted(blank))
            warnings.append(f"Left blank, so the models used a typical training-cohort value (not shown as a patient factor): {labels}.")
        results: List[Dict[str, Any]] = []
        with self._lock:  # tree libraries / SHAP explainers are not guaranteed thread-safe
            for tag in self._model_tags:
                model = models[tag]
                entry = self._entry(tag)
                frame = pd.DataFrame([{c: row.get(c, float("nan")) for c in model.bundle.input_features}])
                prob = float(model.predict_proba(frame)[0, 1])
                explanation = model.explain(frame)
                top = []
                for name in [f for f in explanation["top_features"] if f not in blank][:TOP_N_FACTORS]:
                    value = row.get(name)
                    meta = spec_by_name.get(name, {})
                    value_label = None
                    if value is not None and meta.get("options"):
                        for opt in meta["options"]:
                            if abs(float(opt["value"]) - float(value)) < 1e-9:
                                value_label = str(opt["label"])
                    top.append({
                        "feature": name,
                        "label": meta.get("label", name),
                        "value": None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value),
                        "value_label": value_label,
                        "contribution": round(float(explanation["feature_contributions"][name]), 4),
                        "direction": explanation["feature_directions"][name],
                    })
                ops = self._operating_points(entry, prob)
                selected = operating_point if ops[operating_point]["threshold"] is not None else (
                    "balanced" if ops["balanced"]["threshold"] is not None else "accuracy")
                results.append({
                    "model_id": tag,
                    "model_family": model.model_family,
                    "model_name": DISPLAY_NAME.get(model.model_family, model.model_family),
                    "probability": round(prob, 4),
                    "risk_level": ops[selected]["risk_level"] or ("High" if prob >= model.threshold else "Low"),
                    "threshold": ops[selected]["threshold"] if ops[selected]["threshold"] is not None else round(float(model.threshold), 4),
                    "operating_point": selected,
                    "operating_points": ops,
                    "imbalance_strategy": entry.get("imbalance_strategy"),
                    "top_factors": top,
                    "explanation_summary": explanation["summary"],
                    "test_metrics": entry["test_metrics"],
                    "status": entry.get("status", "pending_review"),
                    "variant": entry.get("variant", ""),
                })
        probs = [r["probability"] for r in results]
        levels = [r["risk_level"] for r in results]
        return {
            "models": results,
            "average_probability": round(float(np.mean(probs)), 4),
            "models_agree": len(set(levels)) == 1,
            "n_high": levels.count("High"),
            "n_low": levels.count("Low"),
            "operating_point": operating_point,
            "n_high_by_point": {pt: sum(1 for r in results if r["operating_points"][pt]["risk_level"] == "High") for pt in OPERATING_POINTS},
            "probability_min": min(probs),
            "probability_max": max(probs),
            "warnings": warnings,
            "metrics_note": METRICS_NOTE,
            "disclaimer": DISCLAIMER,
            "feature_set": self.get_spec().get("primary_variant", ""),
            "inference_time_sec": round(time.perf_counter() - start, 4),
        }


nhanes_compare_service = NhanesCompareService()
