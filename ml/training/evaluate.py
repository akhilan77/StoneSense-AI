"""Evaluation and statistical validation utilities for ML Risk Prediction.

Implements Nested Repeated Stratified K-Fold Cross Validation, Nadeau-Bengio corrected SE,
1-Standard-Error model selection, row-level bootstrap 95% confidence intervals,
and parallel permutation significance tests.
"""

from pathlib import Path
import json
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve, auc
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline

logger = logging.getLogger("MLEvaluator")

MODEL_COMPLEXITY_ORDER = {
    "LogisticRegression": 1,
    "RandomForest": 2,
    "XGBoost": 3
}


def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Computes Sensitivity (TPR / Recall) and Specificity (TNR).
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        
    Returns:
        Tuple of (sensitivity, specificity).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def compute_nadeau_bengio_se(fold_scores: np.ndarray, n_splits: int = 5, n_repeats: int = 10) -> float:
    """Computes Nadeau-Bengio corrected standard error for repeated cross-validation.
    
    Corrects for the correlation between overlapping training sets across repeats.
    Formula: Var_corr = (1 / (J * K) + (1 / K) / (1 - 1 / K)) * s^2
    where J = n_repeats, K = n_splits, s^2 = sample variance across all J * K folds.
    
    Args:
        fold_scores: Array of test scores across all J * K folds.
        n_splits: Number of folds per repeat (K).
        n_repeats: Number of repeats (J).
        
    Returns:
        Corrected standard error.
    """
    n_total = len(fold_scores)
    if n_total <= 1:
        return 0.0
    
    sample_var = float(np.var(fold_scores, ddof=1))
    test_fraction = 1.0 / n_splits
    correction_factor = (1.0 / (n_repeats * n_splits)) + (test_fraction / (1.0 - test_fraction))
    corrected_var = correction_factor * sample_var
    return float(np.sqrt(max(0.0, corrected_var)))


def compute_repeat_means_se(fold_scores: np.ndarray, n_splits: int = 5, n_repeats: int = 10) -> float:
    """Computes standard error of the mean scores across independent CV repeats.
    
    Args:
        fold_scores: Array of test scores across all J * K folds (length = n_repeats * n_splits).
        n_splits: Number of folds per repeat.
        n_repeats: Number of repeats.
        
    Returns:
        Standard error across repeat means.
    """
    scores_matrix = fold_scores.reshape((n_repeats, n_splits))
    repeat_means = np.mean(scores_matrix, axis=1)
    return float(np.std(repeat_means, ddof=1) / np.sqrt(n_repeats))


def evaluate_nested_cv(
    model_factory_or_pipeline: Any,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "roc_auc"
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    """Performs Nested Repeated Stratified K-Fold Cross-Validation.
    
    Outer loop: 5 folds x 10 repeats = 50 evaluations for unbiased assessment.
    Inner loop: 5-fold Stratified GridSearchCV on training fold for hyperparameter selection.
    
    Returns:
        Tuple of (summary_dict, fold_df, oof_probs_matrix, oof_preds_matrix).
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    N = len(y)
    
    oof_probs = np.zeros((n_repeats, N))
    oof_preds = np.zeros((n_repeats, N))
    fold_records: List[Dict[str, Any]] = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        repeat_idx = fold_idx // n_splits
        repeat_num = repeat_idx + 1
        fold_num = (fold_idx % n_splits) + 1
        
        X_train_outer, y_train_outer = X.iloc[train_idx], y[train_idx]
        X_test_outer, y_test_outer = X.iloc[test_idx], y[test_idx]
        
        base_estimator = model_factory_or_pipeline() if callable(model_factory_or_pipeline) else clone(model_factory_or_pipeline)
        
        # Inner CV on outer training fold only (zero leakage to outer test fold)
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + fold_idx)
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=1
        )
        grid_search.fit(X_train_outer, y_train_outer)
        
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Outer fold evaluation
        preds = best_estimator.predict(X_test_outer)
        probs = best_estimator.predict_proba(X_test_outer)[:, 1] if hasattr(best_estimator, "predict_proba") else preds
        
        oof_probs[repeat_idx, test_idx] = probs
        oof_preds[repeat_idx, test_idx] = preds
        
        fold_auc = float(roc_auc_score(y_test_outer, probs))
        fold_f1 = float(f1_score(y_test_outer, preds, zero_division=0))
        fold_mcc = float(matthews_corrcoef(y_test_outer, preds))
        fold_sens, fold_spec = compute_sensitivity_specificity(y_test_outer, preds)
        fold_prec = float(precision_score(y_test_outer, preds, zero_division=0))
        fold_acc = float(accuracy_score(y_test_outer, preds))
        fold_cm = confusion_matrix(y_test_outer, preds, labels=[0, 1]).tolist()
        
        fold_records.append({
            "repeat": repeat_num,
            "fold": fold_num,
            "fold_index": fold_idx + 1,
            "auc": round(fold_auc, 4),
            "f1": round(fold_f1, 4),
            "mcc": round(fold_mcc, 4),
            "sensitivity": round(fold_sens, 4),
            "specificity": round(fold_spec, 4),
            "precision": round(fold_prec, 4),
            "accuracy": round(fold_acc, 4),
            "best_params": best_params,
            "confusion_matrix": fold_cm
        })
        
    df_folds = pd.DataFrame(fold_records)
    aucs = df_folds["auc"].values
    f1s = df_folds["f1"].values
    mccs = df_folds["mcc"].values
    sens = df_folds["sensitivity"].values
    specs = df_folds["specificity"].values
    precs = df_folds["precision"].values
    accs = df_folds["accuracy"].values
    
    auc_se_nb = compute_nadeau_bengio_se(aucs, n_splits=n_splits, n_repeats=n_repeats)
    auc_se_rep = compute_repeat_means_se(aucs, n_splits=n_splits, n_repeats=n_repeats)
    
    summary = {
        "evaluation_strategy": f"Nested RepeatedStratifiedKFold ({n_splits} splits, {n_repeats} repeats = {len(df_folds)} outer folds)",
        "total_outer_folds": len(df_folds),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs, ddof=1)), 4),
        "auc_se_nadeau_bengio": round(auc_se_nb, 4),
        "auc_se_repeats": round(auc_se_rep, 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s, ddof=1)), 4),
        "mcc_mean": round(float(np.mean(mccs)), 4),
        "mcc_std": round(float(np.std(mccs, ddof=1)), 4),
        "sensitivity_mean": round(float(np.mean(sens)), 4),
        "sensitivity_std": round(float(np.std(sens, ddof=1)), 4),
        "specificity_mean": round(float(np.mean(specs)), 4),
        "specificity_std": round(float(np.std(specs, ddof=1)), 4),
        "precision_mean": round(float(np.mean(precs)), 4),
        "precision_std": round(float(np.std(precs, ddof=1)), 4),
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs, ddof=1)), 4),
        "fold_results": fold_records
    }
    
    return summary, df_folds, oof_probs, oof_preds


def compute_row_level_bootstrap_cis(
    y_true: np.ndarray,
    oof_probs_matrix: np.ndarray,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """Computes non-parametric row-level bootstrap 95% confidence intervals.
    
    Averages out-of-fold predicted probabilities per row across repeats,
    then resamples rows with replacement. Each row is treated as an independent patient
    because no patient identifier exists in the clinical dataset.
    
    Calculates bootstrap CIs for:
    - ROC-AUC
    - F1-Score
    - Matthews Correlation Coefficient (MCC)
    - Sensitivity (Recall)
    - Specificity
    """
    from scipy.stats import rankdata
    rng = np.random.RandomState(random_state)
    N = len(y_true)
    
    # 1. Average out-of-fold predicted probabilities per row across repeats
    row_probs = np.mean(oof_probs_matrix, axis=0)
    row_preds = (row_probs >= 0.5).astype(int)
    
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    
    def _fast_auc(y_arr: np.ndarray, scores: np.ndarray) -> float:
        n_pos = np.sum(y_arr == 1)
        n_neg = len(y_arr) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.nan
        ranks = rankdata(scores)
        r_pos = np.sum(ranks[y_arr == 1])
        return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    
    boot_aucs, boot_f1s, boot_mccs, boot_sens, boot_specs = [], [], [], [], []
    boot_precs, boot_accs = [], []
    
    for _ in range(n_bootstraps):
        while True:
            idx = rng.choice(N, size=N, replace=True)
            if len(np.unique(y_true[idx])) > 1:
                break
                
        y_b = y_true[idx]
        probs_b = row_probs[idx]
        preds_b = row_preds[idx]
        
        auc_val = _fast_auc(y_b, probs_b)
        f1_val = float(f1_score(y_b, preds_b, zero_division=0))
        mcc_val = float(matthews_corrcoef(y_b, preds_b))
        sens_val, spec_val = compute_sensitivity_specificity(y_b, preds_b)
        prec_val = float(precision_score(y_b, preds_b, zero_division=0))
        acc_val = float(accuracy_score(y_b, preds_b))
        
        boot_aucs.append(auc_val)
        boot_f1s.append(f1_val)
        boot_mccs.append(mcc_val)
        boot_sens.append(sens_val)
        boot_specs.append(spec_val)
        boot_precs.append(prec_val)
        boot_accs.append(acc_val)
        
    def _format_metric_ci(boot_vals: List[float]) -> Dict[str, Any]:
        arr = np.array(boot_vals)
        arr = arr[~np.isnan(arr)]
        lo = float(np.percentile(arr, lower_pct))
        hi = float(np.percentile(arr, upper_pct))
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=1)), 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "ci_str": f"[{lo:.4f}, {hi:.4f}]"
        }
        
    return {
        "roc_auc": _format_metric_ci(boot_aucs),
        "f1": _format_metric_ci(boot_f1s),
        "mcc": _format_metric_ci(boot_mccs),
        "sensitivity": _format_metric_ci(boot_sens),
        "specificity": _format_metric_ci(boot_specs),
        "precision": _format_metric_ci(boot_precs),
        "accuracy": _format_metric_ci(boot_accs),
        "row_averaged_oof": {
            "roc_auc": round(float(roc_auc_score(y_true, row_probs)), 4),
            "f1": round(float(f1_score(y_true, row_preds, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y_true, row_preds)), 4),
            "sensitivity": round(float(compute_sensitivity_specificity(y_true, row_preds)[0]), 4),
            "specificity": round(float(compute_sensitivity_specificity(y_true, row_preds)[1]), 4),
            "accuracy": round(float(accuracy_score(y_true, row_preds)), 4)
        }
    }


def run_permutation_test(
    model_pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_permutations: int = 1000,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    scoring: str = "roc_auc"
) -> Dict[str, Any]:
    """Executes a non-parametric permutation significance test with fixed hyperparameters.
    
    Evaluates whether the model's predictive power is statistically significantly better
    than chance (null hypothesis: no association between features and labels).
    
    Args:
        model_pipeline: Pipeline with preprocessor and classifier with fixed hyperparameters.
        X: Feature DataFrame.
        y: Target label array.
        n_permutations: Number of label permutations (minimum 1000).
        n_splits: CV splits for evaluation.
        random_state: Random seed.
        n_jobs: Parallel worker count (-1 uses all available cores).
        scoring: Metric to evaluate.
        
    Returns:
        Dictionary containing true score, permutation null scores summary, empirical p-value, and runtime.
    """
    from sklearn.model_selection import permutation_test_score, StratifiedKFold
    
    logger.info(f"Running permutation significance test ({n_permutations} permutations, n_jobs={n_jobs})...")
    start_time = time.time()
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score, perm_scores, pvalue = permutation_test_score(
        estimator=model_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        random_state=random_state
    )
    runtime = time.time() - start_time
    logger.info(f"Permutation test complete in {runtime:.2f}s: true_score={score:.4f}, p_value={pvalue:.4f}")
    
    return {
        "true_cv_score": round(float(score), 4),
        "p_value": round(float(pvalue), 4),
        "perm_scores_mean": round(float(np.mean(perm_scores)), 4),
        "perm_scores_std": round(float(np.std(perm_scores, ddof=1)), 4),
        "n_permutations": n_permutations,
        "runtime_seconds": round(runtime, 2)
    }


def select_model_1se_rule(
    comparison_records: List[Dict[str, Any]],
    primary_metric: str = "ROC-AUC Mean",
    se_metric: str = "Nadeau-Bengio SE",
    baseline_model: str = "LogisticRegression"
) -> Tuple[str, Dict[str, Any]]:
    """Selects the simplest model whose performance is within 1 Standard Error of the best model.
    
    Uses Nadeau-Bengio corrected standard error (or repeat SE) on ROC-AUC.
    
    Complexity hierarchy:
        1: LogisticRegression (Baseline, linear, most interpretable)
        2: RandomForest (Non-linear tree ensemble)
        3: XGBoost (Gradient boosted decision trees)
        
    Args:
        comparison_records: List of dictionaries per model.
        primary_metric: Metric key for performance ranking.
        se_metric: Key for standard error.
        baseline_model: Name of simplest baseline model.
        
    Returns:
        Tuple of (selected_model_name, selection_diagnostics_dict).
    """
    df = pd.DataFrame(comparison_records)
    df["complexity"] = df["Model"].map(MODEL_COMPLEXITY_ORDER).fillna(99)
    
    # Identify top-performing model
    best_row = df.sort_values(by=primary_metric, ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_score = float(best_row[primary_metric])
    best_se = float(best_row[se_metric])
    
    # 1-SE Selection Threshold
    threshold_1se = best_score - best_se
    
    # Models within 1 SE of best
    qualifying_models = df[df[primary_metric] >= threshold_1se].copy()
    qualifying_models = qualifying_models.sort_values(by="complexity", ascending=True)
    
    # Select simplest qualifying model
    selected_model_name = qualifying_models.iloc[0]["Model"]
    
    is_baseline_selected = (selected_model_name == baseline_model)
    is_complex_better_than_1se = (best_model_name != baseline_model and best_score - df[df["Model"] == baseline_model][primary_metric].values[0] > best_se)
    
    diagnostics = {
        "primary_metric": primary_metric,
        "best_performing_model": best_model_name,
        "best_model_mean_score": round(best_score, 4),
        "best_model_se": round(best_se, 4),
        "selection_threshold_1se": round(threshold_1se, 4),
        "qualifying_models_within_1se": qualifying_models["Model"].tolist(),
        "selected_model": selected_model_name,
        "selected_model_complexity_rank": int(MODEL_COMPLEXITY_ORDER.get(selected_model_name, 99)),
        "is_baseline_selected": is_baseline_selected,
        "is_any_model_significantly_better_than_baseline": bool(is_complex_better_than_1se)
    }
    
    logger.info(
        f"1-SE Model Selection Result: Best={best_model_name} ({best_score:.4f} ± {best_se:.4f}), "
        f"Threshold={threshold_1se:.4f} --> Selected: {selected_model_name} (Complexity Rank: {diagnostics['selected_model_complexity_rank']})"
    )
    return selected_model_name, diagnostics


def evaluate_ml_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str]
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Computes comprehensive evaluation metrics on a test dataset."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probs))
    mcc = float(matthews_corrcoef(y_test, preds))
    sens, spec = compute_sensitivity_specificity(y_test, preds)

    cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "matthews_correlation_coefficient": round(mcc, 4),
        "confusion_matrix": cm
    }

    return metrics, preds, probs


def save_ml_charts(
    y_test: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    charts_dir: Path
) -> None:
    """Saves confusion matrix heatmap, ROC curves, and PR curves."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Urine Analysis Risk - Confusion Matrix", fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("True Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(charts_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#059669", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(charts_dir / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#2563eb", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall", fontweight="bold")
    plt.ylabel("Precision", fontweight="bold")
    plt.title("Precision-Recall Curve", fontweight="bold", pad=15)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(charts_dir / "precision_recall_curve.png", dpi=300)
    plt.close()
