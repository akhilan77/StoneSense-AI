"""Model Hyperparameter Optimization space definitions for Kidney Stone Risk prediction.

Configures search grids for Logistic Regression, Random Forest, and XGBoost.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Models dictionary
MODELS: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Parameter search grids
PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, None],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2]
    }
}
