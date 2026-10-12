# StoneSense-AI Machine Learning Subsystem (Tabular Risk)

The `ml/` directory contains the data validation, preprocessing, nested cross-validation training, statistical evaluation, dynamic SHAP explainability, and model version registry for urinary biomarker risk prediction.

---

## Directory Layout

```text
ml/
├── artifacts/                # Preprocessing and scaling artifacts
│   ├── feature_columns.pkl   # Authoritative list of 6 biomarker features
│   ├── preprocessing_metadata.json # Metadata and feature order
│   ├── preprocessing_pipeline.pkl  # Sklearn preprocessing pipeline
│   └── scaler.pkl            # Standard scaler artifact
├── explainability/
│   └── shap_explainer.py     # Dynamic SHAP explainer (LinearExplainer & TreeExplainer)
├── models/                   # Versioned models, manifests, and evaluation files
│   ├── versions/             # Individual versioned model pickles
│   │   ├── logistic_regression_v001.pkl  # Deployed 1-SE rule winner
│   │   └── xgboost_v001.pkl              # Registered candidate
│   ├── candidate_risk_model.pkl          # Active candidate pickle
│   ├── kidney_risk_model.pkl             # Standalone model artifact (XGBoost)
│   ├── comparison_results.csv            # 50-fold nested CV comparative metrics
│   ├── cv_fold_metrics.csv               # Metrics per fold across 50 iterations
│   ├── cv_results.json                   # Comprehensive cross-validation JSON
│   ├── feature_importance.json           # Global feature importance weights
│   └── registry_manifest.json            # Registry manifest with active pointer
├── preprocessing/            # Tabular data integrity and transformations
│   ├── tabular_preprocessor.py # Duplicate detection and class balance verification
│   ├── feature_engineering.py  # Ratio and biomarker feature transformers
│   ├── split_dataset.py        # Stratified train/test splitter
│   └── validate_ml_dataset.py  # Integrity validation CLI
├── risk_models/              # Abstract BaseRiskModel and Registry architecture
│   ├── base.py               # Abstract BaseRiskModel class
│   ├── registry.py           # ModelRegistry managing versions and manifests
│   └── wrappers.py           # Class wrappers for LogisticRegression, RandomForest, XGBoost
└── training/                 # Cross-validation training and statistical evaluation
    ├── evaluate.py           # Bootstrap CIs, permutation tests, 1-SE selection rule
    ├── model.py              # Estimators and hyperparameter grids
    └── train_risk_model.py   # Main 50-fold nested CV training script
```

---

## Model Selection & Verified Metrics

The dataset consists of **79 unique patient observations** with 6 urinary biomarkers (`gravity`, `ph`, `osmo`, `cond`, `urea`, `calc`).

The pipeline runs **5 folds $\times$ 10 repeats (50 total folds)** nested cross-validation with row-level bootstrap confidence intervals (1,000 samples) and permutation testing (1,000 iterations).

### 50-Fold Nested CV Summary (Source: [`ml/models/comparison_results.csv`](models/comparison_results.csv))

| Model | Complexity | Mean ROC-AUC (95% CI) | Mean Macro F1 (95% CI) | Mean Accuracy (95% CI) | Permutation $p$ | Selection |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **1** | **0.8387** `[0.7327, 0.9176]` | **0.6937** `[0.5333, 0.7937]` | **74.67%** `[0.6329, 0.8231]` | **0.001** | **1-SE WINNER (ACTIVE)** |
| **Random Forest** | 2 | 0.8175 `[0.7122, 0.9191]` | 0.7067 `[0.5778, 0.8302]` | 75.62% `[0.6835, 0.8608]` | 0.001 | Alternative Candidate |
| **XGBoost** | 3 | 0.8076 `[0.7080, 0.9156]` | 0.7169 `[0.5862, 0.8308]` | 77.14% `[0.6835, 0.8608]` | 0.001 | Alternative Candidate |

Under the **1-Standard-Error rule**, Logistic Regression achieved the highest mean ROC-AUC while having the lowest structural complexity, and was designated the active deployed model in [`ml/models/registry_manifest.json`](models/registry_manifest.json).

---

## How to Retrain & Rerun ML Evaluation

```bash
# Run 50-fold nested repeated cross validation:
python ml/training/train_risk_model.py

# Generate SHAP waterfall and summary plots:
python ml/explainability/shap_explainer.py
```
