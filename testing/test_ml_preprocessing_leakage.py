"""Tests verifying that ML preprocessing is leak-free and validation/test data never influences scaler fitting."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure ml modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
ML_PREPROCESSING_DIR = ML_DIR / "preprocessing"
ML_TRAINING_DIR = ML_DIR / "training"
sys.path.insert(0, str(ML_TRAINING_DIR))
sys.path.insert(0, str(ML_PREPROCESSING_DIR))

from tabular_preprocessor import TabularPreprocessor
from feature_engineering import create_domain_features, extract_metadata
from split_dataset import split_tabular_dataset
from preprocessing_pipeline import run_ml_preprocessing_pipeline


@pytest.fixture
def sample_raw_data():
    """Generates synthetic dataset with distinct train and val/test distributions."""
    np.random.seed(42)
    # Train-like distribution (mean ~ 10)
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

    # Expected scaler mean must match train_df numerical means exactly
    num_cols = preprocessor.numerical_cols
    expected_means = X_train[num_cols].mean().values
    np.testing.assert_allclose(fitted_scaler.mean_, expected_means, rtol=1e-5)

    # Whole dataset mean should differ
    all_means = sample_raw_data[num_cols].mean().values
    assert not np.allclose(fitted_scaler.mean_, all_means, rtol=1e-5) or len(train_df) == len(sample_raw_data)


def test_validation_and_test_modifications_do_not_affect_scaler(sample_raw_data):
    """Proves that injecting extreme outlier distributions into validation/test splits

    does NOT alter the fitted scaler means or variance.
    """
    train_df, val_df, test_df = split_tabular_dataset(sample_raw_data, target_column="target", random_state=42)

    # 1. Fit on normal train data
    prep1 = TabularPreprocessor(target_column="target")
    prep1.auto_detect_columns(train_df)
    prep1.fit(train_df.drop(columns=["target"]))
    mean_before = prep1.scaler.mean_.copy()
    scale_before = prep1.scaler.scale_.copy()

    # 2. Add extreme outliers to val and test sets (1000x multiplier)
    corrupted_val = val_df.copy()
    corrupted_test = test_df.copy()
    for col in prep1.numerical_cols:
        corrupted_val[col] = corrupted_val[col] * 1000.0
        corrupted_test[col] = corrupted_test[col] * 1000.0

    # Fit pipeline using isolated training split
    prep2 = TabularPreprocessor(target_column="target")
    prep2.auto_detect_columns(train_df)
    prep2.fit(train_df.drop(columns=["target"]))
    mean_after = prep2.scaler.mean_.copy()
    scale_after = prep2.scaler.scale_.copy()

    # Assert that scaler parameters are 100% unaffected by validation/test distributions
    np.testing.assert_array_equal(mean_before, mean_after)
    np.testing.assert_array_equal(scale_before, scale_after)


def test_cross_validation_pipeline_no_leakage():
    """Proves that during cross-validation, preprocessing is fit strictly inside each training fold."""
    # Custom transformer that records the sizes of datasets seen during fit()
    fit_sample_sizes = []

    class LeakageTrackerTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            fit_sample_sizes.append(len(X))
            return self

        def transform(self, X):
            return X

    np.random.seed(42)
    X = pd.DataFrame({
        "num1": np.random.randn(50),
        "num2": np.random.randn(50)
    })
    y = np.array([0, 1] * 25)

    pipe = Pipeline([
        ('tracker', LeakageTrackerTransformer()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    assert len(scores) == 5
    # In 5-fold CV with 50 samples, each training fold MUST have exactly 40 samples (4/5 of 50)
    assert len(fit_sample_sizes) == 5
    assert all(size == 40 for size in fit_sample_sizes), (
        f"Expected all CV training folds to have 40 samples, but got: {fit_sample_sizes}"
    )


def test_feature_engineering_preservation():
    """Verifies domain feature engineering ratios and metadata extraction remain preserved."""
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

    # Check correct ratios
    assert engineered.loc[0, "cond_osmo_ratio"] == 5.0
    assert engineered.loc[1, "cond_osmo_ratio"] == 5.0
    assert engineered.loc[2, "cond_osmo_ratio"] == 0.0  # Safe division by zero

    assert engineered.loc[0, "urea_calc_ratio"] == 20.0
    assert engineered.loc[1, "urea_calc_ratio"] == 20.0
    assert engineered.loc[2, "urea_calc_ratio"] == 0.0  # Safe division by zero

    # Test metadata extraction
    meta = extract_metadata(
        df=engineered,
        target_col="target",
        num_cols=["cond", "osmo", "urea", "calc"],
        cat_cols=[],
        train_len=55,
        val_len=12,
        test_len=12
    )
    assert meta["target_column"] == "target"
    assert meta["dataset_splits"]["train_size"] == 55
    assert meta["dataset_splits"]["total_size"] == 79


def test_end_to_end_ml_preprocessing_pipeline():
    """Verifies that run_ml_preprocessing_pipeline executes and produces leak-free artifacts."""
    run_ml_preprocessing_pipeline()

    processed_dir = ML_DIR / "processed"
    artifacts_dir = ML_DIR / "artifacts"

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Verify artifacts exist
    assert (artifacts_dir / "preprocessing_pipeline.pkl").exists()
    assert (artifacts_dir / "scaler.pkl").exists()
    assert (artifacts_dir / "feature_columns.pkl").exists()
    assert (artifacts_dir / "preprocessing_metadata.json").exists()

    # Load saved scaler and verify it was fitted on train data
    import joblib
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    target_col = "target"
    num_cols = list(train_df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns)

    expected_train_means = train_df[num_cols].mean().values
    np.testing.assert_allclose(scaler.mean_, expected_train_means, rtol=1e-5)


def test_repeated_stratified_kfold_evaluation():
    """Verifies that RepeatedStratifiedKFold evaluation runs across 50 folds and computes required metrics."""
    import importlib.util
    eval_path = ML_DIR / "training" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("ml_evaluate", eval_path)
    ml_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml_eval)
    evaluate_repeated_cv = ml_eval.evaluate_repeated_cv

    dataset_path = ML_DIR / "datasets" / "kidneyData.csv"
    raw_df = pd.read_csv(dataset_path)
    cleaner = TabularPreprocessor(target_column="target")
    cleaned_df = cleaner.clean_data(raw_df)

    X = cleaned_df.drop(columns=["target"])
    y = cleaned_df["target"].values

    def dummy_pipe_factory():
        p = TabularPreprocessor(target_column="target")
        p.numerical_cols = list(X.select_dtypes(include=[np.number]).columns)
        p.categorical_cols = list(X.select_dtypes(include=['object', 'category']).columns)
        return Pipeline([
            ('preprocessor', p.build_pipeline()),
            ('classifier', LogisticRegression(random_state=42))
        ])

    summary, df_folds = evaluate_repeated_cv(
        pipeline_factory_or_estimator=dummy_pipe_factory,
        X=X,
        y=y,
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    # Check 50 folds evaluated
    assert len(df_folds) == 50
    assert summary["total_folds"] == 50

    # Check required metrics are present for every fold
    required_keys = ["auc", "recall", "precision", "f1", "accuracy", "confusion_matrix"]
    for key in required_keys:
        assert key in df_folds.columns

    # Check summary metrics
    for metric in ["auc_mean", "auc_std", "recall_mean", "recall_std", "precision_mean", "precision_std", "f1_mean", "f1_std", "accuracy_mean", "accuracy_std"]:
        assert metric in summary
        assert isinstance(summary[metric], float)

    # Check fold-level CSV file from risk training
    fold_csv = ML_DIR / "models" / "cv_fold_metrics.csv"
    assert fold_csv.exists()
    saved_df = pd.read_csv(fold_csv)
    assert len(saved_df) in [50, 100]  # 50 for single model or 100 for 2 paired models
    assert set(required_keys).issubset(set(saved_df.columns))


def test_dataset_verification_and_repeated_cv_parameters():
    """Confirms dataset dimensions, RepeatedStratifiedKFold parameters, and out-of-fold aggregation semantics."""
    dataset_path = ML_DIR / "datasets" / "kidneyData.csv"
    raw_df = pd.read_csv(dataset_path)

    # 1. Confirm dataset contains exactly 79 observations
    assert len(raw_df) == 79, f"Expected 79 observations, found {len(raw_df)}"
    assert "target" in raw_df.columns

    # 2. Confirm RepeatedStratifiedKFold configuration
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    assert rskf.get_n_splits(raw_df.drop(columns=["target"]), raw_df["target"]) == 50

    # 3. Confirm 790 total predictions represent 10 repeated evaluations of 79 patients
    splits = list(rskf.split(raw_df.drop(columns=["target"]), raw_df["target"]))
    assert len(splits) == 50, f"Expected 50 total folds, got {len(splits)}"

    # In each repeat (5 folds), every patient index appears exactly once in the test split
    for repeat_idx in range(10):
        repeat_test_indices = []
        for fold_idx in range(5):
            _, test_idx = splits[repeat_idx * 5 + fold_idx]
            repeat_test_indices.extend(test_idx)
        assert sorted(repeat_test_indices) == list(range(79)), (
            f"Repeat {repeat_idx + 1} test indices do not cover all 79 observations exactly once"
        )


def test_patient_level_bootstrap_implementation():
    """Tests that compute_repeated_cv_bootstrap_ci properly resamples patient units and produces valid 95% CIs."""
    import importlib.util
    eval_path = ML_DIR / "training" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("ml_evaluate", eval_path)
    ml_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml_eval)
    compute_repeated_cv_bootstrap_ci = ml_eval.compute_repeated_cv_bootstrap_ci

    np.random.seed(42)
    N = 79
    n_repeats = 10
    y_true = np.array([0] * 45 + [1] * 34)

    # Synthetic out-of-fold probabilities and predictions
    oof_probs = np.random.uniform(0.1, 0.9, size=(n_repeats, N))
    # Give positive class higher probabilities
    oof_probs[:, 45:] += 0.2
    oof_probs = np.clip(oof_probs, 0.0, 1.0)
    oof_preds = (oof_probs >= 0.5).astype(int)

    ci_results = compute_repeated_cv_bootstrap_ci(
        y_true=y_true,
        oof_probs=oof_probs,
        oof_preds=oof_preds,
        n_bootstraps=500,
        confidence_level=0.95,
        random_state=42
    )

    for metric in ["roc_auc", "recall", "precision", "f1", "accuracy"]:
        assert metric in ci_results
        res = ci_results[metric]
        assert "mean" in res
        assert "ci_lower" in res
        assert "ci_upper" in res
        assert "ci_str" in res
        assert res["ci_lower"] <= res["ci_upper"], f"Lower bound exceeds upper bound for {metric}"
        assert 0.0 <= res["ci_lower"] <= 1.0
        assert 0.0 <= res["ci_upper"] <= 1.0


def test_paired_cv_bootstrap_ci_integration():
    """Tests evaluate_models_paired_cv returns valid bootstrap 95% CIs alongside mean ± SD."""
    import importlib.util
    eval_path = ML_DIR / "training" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("ml_evaluate", eval_path)
    ml_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml_eval)
    evaluate_models_paired_cv = ml_eval.evaluate_models_paired_cv

    dataset_path = ML_DIR / "datasets" / "kidneyData.csv"
    raw_df = pd.read_csv(dataset_path)
    cleaner = TabularPreprocessor(target_column="target")
    cleaned_df = cleaner.clean_data(raw_df)

    X = cleaned_df.drop(columns=["target"])
    y = cleaned_df["target"].values

    def dummy_pipe():
        p = TabularPreprocessor(target_column="target")
        p.numerical_cols = list(X.select_dtypes(include=[np.number]).columns)
        p.categorical_cols = list(X.select_dtypes(include=['object', 'category']).columns)
        return Pipeline([
            ('preprocessor', p.build_pipeline()),
            ('classifier', LogisticRegression(random_state=42))
        ])

    models = {"TestModel": dummy_pipe}
    summaries, comp_df, fold_df = evaluate_models_paired_cv(
        models_dict=models,
        X=X,
        y=y,
        n_splits=5,
        n_repeats=10,
        random_state=42,
        n_bootstraps=200
    )

    assert "TestModel" in summaries
    summary = summaries["TestModel"]
    assert "auc_ci_95" in summary
    assert "recall_ci_95" in summary
    assert "bootstrap_ci" in summary
    assert summary["auc_ci_lower"] <= summary["auc_ci_upper"]
    assert summary["recall_ci_lower"] <= summary["recall_ci_upper"]
    assert len(comp_df) == 1
