"""Model Hyperparameter Optimization space definitions for Kidney Stone Risk prediction.

Configures classifiers and search grids for Logistic Regression, Random Forest, and XGBoost.
SMOTE and synthetic oversampling are strictly prohibited.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    xgb_clf = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1)
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    xgb_clf = GradientBoostingClassifier(random_state=42)

MODEL_COMPLEXITY_ORDER: Dict[str, int] = {
    "LogisticRegression": 1,
    "RandomForest": 2,
    "XGBoost": 3
}

# Base models dictionary
MODELS: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1),
    "XGBoost": xgb_clf
}

# Parameter search grids for inner loop of Nested CV (curated for N~80 dataset)
PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    "RandomForest": {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_split": [2, 5]
    },
    "XGBoost": {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1]
    }
}
