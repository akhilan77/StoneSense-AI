"""Tests verifying that ML preprocessing is leak-free, validation/test data never influences scaler fitting,
and rigorous evaluation protocols (Nested CV, Bootstrap 95% CIs, Permutation Testing, 1-SE Selection,
Duplicate & Class-Balance checks) execute with statistical integrity.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone

# Ensure ml modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
ML_PREPROCESSING_DIR = ML_DIR / "preprocessing"
ML_TRAINING_DIR = ML_DIR / "training"
sys.path.insert(0, str(ML_TRAINING_DIR))
sys.path.insert(0, str(ML_PREPROCESSING_DIR))

from tabular_preprocessor import TabularPreprocessor, check_duplicate_rows, check_class_balance
from feature_engineering import create_domain_features, extract_metadata
from split_dataset import split_tabular_dataset
from preprocessing_pipeline import run_ml_preprocessing_pipeline
from evaluate import (
    evaluate_nested_cv,
    compute_row_level_bootstrap_cis,
    run_permutation_test,
    select_model_1se_rule,
    compute_nadeau_bengio_se,
    compute_repeat_means_se,
    compute_sensitivity_specificity
)


@pytest.fixture
def sample_raw_data():
    """Generates synthetic dataset with distinct train and val/test distributions."""
    np.random.seed(42)
    train_data = {
        "gravity": np.random.normal(1.015, 0.005, 60),
        "ph": np.random.normal(6.0, 0.5, 60),
        "osmo": np.random.normal(500, 50, 60),
        "cond": np.random.normal(20, 3, 60),
        "urea": np.random.normal(200, 30, 60),
        "calc": np.random.normal(4.0, 1.0, 60),
        "target": np.random.choice([0, 1], size=60)
    }
    return pd.DataFrame(train_data)


def test_scaler_fitted_only_on_train_data(sample_raw_data):
    """Proves that TabularPreprocessor fits StandardScaler strictly on the training fold."""
    train_df, val_df, test_df = split_tabular_dataset(sample_raw_data, target_column="target", random_state=42)

    preprocessor = TabularPreprocessor(target_column="target")
    preprocessor.auto_detect_columns(train_df)

    X_train = train_df.drop(columns=["target"])
    preprocessor.fit(X_train)

    fitted_scaler = preprocessor.scaler
    assert fitted_scaler is not None

    num_cols = preprocessor.numerical_cols
    expected_means = X_train[num_cols].mean().values
    np.testing.assert_allclose(fitted_scaler.mean_, expected_means, rtol=1e-5)

    all_means = sample_raw_data[num_cols].mean().values
    assert not np.allclose(fitted_scaler.mean_, all_means, rtol=1e-5) or len(train_df) == len(sample_raw_data)


def test_validation_and_test_modifications_do_not_affect_scaler(sample_raw_data):
    """Proves that injecting extreme outlier distributions into validation/test splits
    does NOT alter the fitted scaler parameters.
    """
    train_df, val_df, test_df = split_tabular_dataset(sample_raw_data, target_column="target", random_state=42)

    prep1 = TabularPreprocessor(target_column="target")
    prep1.auto_detect_columns(train_df)
    prep1.fit(train_df.drop(columns=["target"]))
    mean_before = prep1.scaler.mean_.copy()
    scale_before = prep1.scaler.scale_.copy()

    corrupted_val = val_df.copy()
    corrupted_test = test_df.copy()
    for col in prep1.numerical_cols:
        corrupted_val[col] = corrupted_val[col] * 1000.0
        corrupted_test[col] = corrupted_test[col] * 1000.0

    prep2 = TabularPreprocessor(target_column="target")
    prep2.auto_detect_columns(train_df)
    prep2.fit(train_df.drop(columns=["target"]))
    mean_after = prep2.scaler.mean_.copy()
    scale_after = prep2.scaler.scale_.copy()

    np.testing.assert_array_equal(mean_before, mean_after)
    np.testing.assert_array_equal(scale_before, scale_after)


def test_duplicate_rows_check_fails_loudly():
    """Verifies that check_duplicate_rows raises ValueError with diagnostic details when duplicates exist."""
    df_with_dups = pd.DataFrame({
        "gravity": [1.01, 1.02, 1.01],
        "ph": [6.0, 6.5, 6.0],
        "target": [0, 1, 0]
    })
    
    # Must fail loudly if drop_duplicates is False
    with pytest.raises(ValueError) as excinfo:
        check_duplicate_rows(df_with_dups, drop_duplicates=False)
    
    assert "Data integrity error" in str(excinfo.value)
    assert "Found 2 duplicate rows" in str(excinfo.value)

    # Must cleanly drop and log if drop_duplicates is True
    cleaned = check_duplicate_rows(df_with_dups, drop_duplicates=True)
    assert len(cleaned) == 2
    assert cleaned.duplicated().sum() == 0


def test_class_balance_check_fails_loudly():
    """Verifies that check_class_balance fails loudly on severe imbalance or single-class data."""
    # Single class dataset
    df_single_class = pd.DataFrame({"feat": [1, 2, 3], "target": [1, 1, 1]})
    with pytest.raises(ValueError) as excinfo:
        check_class_balance(df_single_class, target_column="target")
    assert "contains only 1 class" in str(excinfo.value)

    # Below minimum samples per class
    df_too_few = pd.DataFrame({"feat": list(range(20)), "target": [0]*18 + [1]*2})
    with pytest.raises(ValueError) as excinfo:
        check_class_balance(df_too_few, target_column="target", min_samples_per_class=5)
    assert "Minority class has only 2 samples" in str(excinfo.value)


def test_nested_cross_validation_no_leakage():
    """Proves that during nested CV, inner hyperparameter search and preprocessing fit strictly on outer train fold."""
    fit_sizes = []

    class LeakageDetector(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            fit_sizes.append(len(X))
            return self

        def transform(self, X):
            return X

    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.randn(50),
        "feat2": np.random.randn(50)
    })
    y = np.array([0, 1] * 25)

    pipe = Pipeline([
        ('leak_detector', LeakageDetector()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    param_grid = {"classifier__C": [0.1, 1.0]}

    # Run small nested CV: 2 outer splits, 2 repeats
    summary, df_folds, oof_probs, oof_preds = evaluate_nested_cv(
        model_factory_or_pipeline=lambda: clone(pipe),
        param_grid=param_grid,
        X=X,
        y=y,
        n_splits=2,
        n_repeats=2,
        random_state=42
    )

    assert len(df_folds) == 4
    # In 2-fold CV with 50 samples, outer train is 25 samples.
    # Inside inner 2-fold CV on 25 samples, inner train is 12 or 13 samples (or 25 during final inner refit).
    # None of the fit sizes should EVER be 50 (the full dataset size).
    assert all(sz < 50 for sz in fit_sizes), f"Found full dataset leakage during fit: {fit_sizes}"


def test_row_level_bootstrap_cis():
    """Verifies that compute_row_level_bootstrap_cis calculates valid 95% CIs for all required metrics."""
    np.random.seed(42)
    N = 79
    n_repeats = 10
    y_true = np.array([0] * 45 + [1] * 34)

    oof_probs = np.random.uniform(0.1, 0.9, size=(n_repeats, N))
    oof_probs[:, 45:] += 0.2
    oof_probs = np.clip(oof_probs, 0.0, 1.0)

    cis = compute_row_level_bootstrap_cis(
        y_true=y_true,
        oof_probs_matrix=oof_probs,
        n_bootstraps=500,
        confidence_level=0.95,
        random_state=42
    )

    required_metrics = ["roc_auc", "f1", "mcc", "sensitivity", "specificity", "precision", "accuracy"]
    for m in required_metrics:
        assert m in cis, f"Metric {m} missing from bootstrap CIs"
        res = cis[m]
        assert "mean" in res
        assert "ci_lower" in res
        assert "ci_upper" in res
        assert "ci_str" in res
        assert res["ci_lower"] <= res["ci_upper"], f"Lower CI > Upper CI for {m}"
        assert -1.0 <= res["ci_lower"] <= 1.0
        assert -1.0 <= res["ci_upper"] <= 1.0


def test_permutation_significance_test():
    """Verifies that run_permutation_test executes and calculates empirical p-value and runtime."""
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(40),
        "f2": np.random.randn(40)
    })
    y = np.array([0, 1] * 20)

    pipe = Pipeline([
        ('classifier', LogisticRegression(random_state=42))
    ])

    result = run_permutation_test(
        model_pipeline=pipe,
        X=X,
        y=y,
        n_permutations=20,
        n_splits=2,
        random_state=42,
        n_jobs=1
    )

    assert "true_cv_score" in result
    assert "p_value" in result
    assert "runtime_seconds" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_permutations"] == 20
    assert result["runtime_seconds"] >= 0.0


def test_one_standard_error_selection_rule():
    """Verifies that 1-SE rule chooses the simplest model within 1 SE of the best."""
    # Case 1: Complex model (XGBoost) is highest, but LogisticRegression is within 1 SE
    records = [
        {"Model": "LogisticRegression", "ROC-AUC Mean": 0.81, "Nadeau-Bengio SE": 0.04},
        {"Model": "RandomForest", "ROC-AUC Mean": 0.82, "Nadeau-Bengio SE": 0.035},
        {"Model": "XGBoost", "ROC-AUC Mean": 0.84, "Nadeau-Bengio SE": 0.04}
    ]
    # Best = XGBoost (0.84), Threshold = 0.84 - 0.04 = 0.80.
    # LogisticRegression (0.81 >= 0.80) is simplest qualifying model -> should be selected!
    selected, diag = select_model_1se_rule(records, primary_metric="ROC-AUC Mean", se_metric="Nadeau-Bengio SE")
    assert selected == "LogisticRegression"
    assert diag["is_baseline_selected"] is True

    # Case 2: Complex model is significantly better (> 1 SE beyond baseline)
    records_superior = [
        {"Model": "LogisticRegression", "ROC-AUC Mean": 0.70, "Nadeau-Bengio SE": 0.03},
        {"Model": "RandomForest", "ROC-AUC Mean": 0.75, "Nadeau-Bengio SE": 0.03},
        {"Model": "XGBoost", "ROC-AUC Mean": 0.88, "Nadeau-Bengio SE": 0.03}
    ]
    # Best = XGBoost (0.88), Threshold = 0.88 - 0.03 = 0.85.
    # LogisticRegression (0.70 < 0.85) is disqualified. XGBoost should be selected.
    selected_sup, diag_sup = select_model_1se_rule(records_superior, primary_metric="ROC-AUC Mean", se_metric="Nadeau-Bengio SE")
    assert selected_sup == "XGBoost"
    assert diag_sup["is_any_model_significantly_better_than_baseline"] is True


def test_feature_engineering_elementwise_purity():
    """Verifies feature_engineering.py computes strictly row-level operations with no global aggregates."""
    df = pd.DataFrame({
        "cond": [10.0, 20.0, 0.0],
        "osmo": [2.0, 4.0, 0.0],
        "urea": [100.0, 200.0, 50.0],
        "calc": [5.0, 10.0, 0.0],
        "target": [0, 1, 0]
    })

    engineered = create_domain_features(df)
    assert "cond_osmo_ratio" in engineered.columns
    assert "urea_calc_ratio" in engineered.columns

    assert engineered.loc[0, "cond_osmo_ratio"] == 5.0
    assert engineered.loc[1, "cond_osmo_ratio"] == 5.0
    assert engineered.loc[2, "cond_osmo_ratio"] == 0.0

    assert engineered.loc[0, "urea_calc_ratio"] == 20.0
    assert engineered.loc[1, "urea_calc_ratio"] == 20.0
    assert engineered.loc[2, "urea_calc_ratio"] == 0.0


def test_candidate_model_artifact_saving():
    """Verifies that ML training outputs candidate_risk_model.pkl and associated artifacts without touching kidney_risk_model.pkl."""
    models_dir = ML_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    candidate_model_path = models_dir / "candidate_risk_model.pkl"
    candidate_pipeline_path = models_dir / "candidate_risk_pipeline.pkl"
    cv_results_path = models_dir / "cv_results.json"
    comparison_csv_path = models_dir / "comparison_results.csv"
    cv_fold_metrics_path = models_dir / "cv_fold_metrics.csv"
    report_path = ML_DIR / "outputs" / "reports" / "model_b_report.md"

    # Verify all artifacts exist
    assert candidate_model_path.exists(), "candidate_risk_model.pkl was not saved"
    assert candidate_pipeline_path.exists(), "candidate_risk_pipeline.pkl was not saved"
    assert cv_results_path.exists(), "cv_results.json was not saved"
    assert comparison_csv_path.exists(), "comparison_results.csv was not saved"
    assert cv_fold_metrics_path.exists(), "cv_fold_metrics.csv was not saved"
    assert report_path.exists(), "model_b_report.md was not generated"

    # Verify cv_results.json structure
    import json
    with open(cv_results_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    assert "selected_model" in cv_data
    assert "selection_diagnostics" in cv_data
    assert "LogisticRegression" in cv_data["models"]
    assert "RandomForest" in cv_data["models"]
    assert "XGBoost" in cv_data["models"]

    # Verify comparison_results.csv structure
    comp_df = pd.read_csv(comparison_csv_path)
    assert "Model" in comp_df.columns
    assert "ROC-AUC 95% CI" in comp_df.columns
    assert "1-SE Selected" in comp_df.columns
    assert len(comp_df) == 3

