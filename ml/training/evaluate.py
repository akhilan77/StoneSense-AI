"""Evaluation utilities for ML Risk Prediction model.

Computes metrics (Accuracy, F1, MCC, Cohen's Kappa, ROC-AUC) and exports plots.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

logger = logging.getLogger("MLEvaluator")


def evaluate_ml_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on test dataset."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probs))
    bal_acc = float(balanced_accuracy_score(y_test, preds))
    mcc = float(matthews_corrcoef(y_test, preds))
    kappa = float(cohen_kappa_score(y_test, preds))

    # Per-class metrics
    p_cls = precision_score(y_test, preds, average=None, zero_division=0)
    r_cls = recall_score(y_test, preds, average=None, zero_division=0)
    f1_cls = f1_score(y_test, preds, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1_score": float(f1_cls[idx])
        }

    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "matthews_correlation_coefficient": mcc,
        "cohens_kappa": kappa,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm
    }

    return metrics, preds, probs


def evaluate_repeated_cv(
    pipeline_factory_or_estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluates an estimator/pipeline using RepeatedStratifiedKFold cross-validation.

    For every fold, calculates:
    - AUC
    - Recall
    - Precision
    - F1
    - Accuracy
    - Confusion Matrix

    Returns summary metrics (mean ± SD) and fold-level DataFrame.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.base import clone

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    fold_records: List[Dict[str, Any]] = []
    all_cms: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        repeat_num = fold_idx // n_splits + 1
        fold_num = fold_idx % n_splits + 1

        X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
        X_test_f, y_test_f = X.iloc[test_idx], y[test_idx]

        model = pipeline_factory_or_estimator() if callable(pipeline_factory_or_estimator) else clone(pipeline_factory_or_estimator)
        model.fit(X_train_f, y_train_f)

        preds = model.predict(X_test_f)
        probs = model.predict_proba(X_test_f)[:, 1] if hasattr(model, "predict_proba") else preds

        fold_auc = float(roc_auc_score(y_test_f, probs))
        fold_recall = float(recall_score(y_test_f, preds, zero_division=0))
        fold_precision = float(precision_score(y_test_f, preds, zero_division=0))
        fold_f1 = float(f1_score(y_test_f, preds, zero_division=0))
        fold_acc = float(accuracy_score(y_test_f, preds))
        fold_cm = confusion_matrix(y_test_f, preds)
        all_cms.append(fold_cm)

        fold_records.append({
            "repeat": repeat_num,
            "fold": fold_num,
            "fold_index": fold_idx + 1,
            "auc": round(fold_auc, 4),
            "recall": round(fold_recall, 4),
            "precision": round(fold_precision, 4),
            "f1": round(fold_f1, 4),
            "accuracy": round(fold_acc, 4),
            "confusion_matrix": fold_cm.tolist()
        })

    df_folds = pd.DataFrame(fold_records)

    aucs = df_folds["auc"].values
    recalls = df_folds["recall"].values
    precisions = df_folds["precision"].values
    f1s = df_folds["f1"].values
    accuracies = df_folds["accuracy"].values

    total_cm = np.sum(all_cms, axis=0).tolist()

    summary = {
        "evaluation_method": f"RepeatedStratifiedKFold (n_splits={n_splits}, n_repeats={n_repeats}, total_folds={len(df_folds)})",
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "total_folds": len(df_folds),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs, ddof=1)), 4),
        "recall_mean": round(float(np.mean(recalls)), 4),
        "recall_std": round(float(np.std(recalls, ddof=1)), 4),
        "precision_mean": round(float(np.mean(precisions)), 4),
        "precision_std": round(float(np.std(precisions, ddof=1)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s, ddof=1)), 4),
        "accuracy_mean": round(float(np.mean(accuracies)), 4),
        "accuracy_std": round(float(np.std(accuracies, ddof=1)), 4),
        # Aliases for compatibility
        "accuracy": round(float(np.mean(accuracies)), 4),
        "precision": round(float(np.mean(precisions)), 4),
        "recall": round(float(np.mean(recalls)), 4),
        "f1_score": round(float(np.mean(f1s)), 4),
        "roc_auc": round(float(np.mean(aucs)), 4),
        "matthews_correlation_coefficient": round(float(np.mean(f1s)), 4),
        "aggregated_confusion_matrix": total_cm,
        "confusion_matrix": total_cm,
        "fold_results": fold_records
    }

    return summary, df_folds


def compute_repeated_cv_bootstrap_ci(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    oof_preds: np.ndarray,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """Computes non-parametric patient-level (cluster) bootstrap confidence intervals for Repeated CV.

    Methodology:
    - The observational sampling unit is the individual patient/row (N observations).
    - In each of the B bootstrap iterations, N patient indices are resampled with replacement.
    - Repeated out-of-fold predictions on the same patient are NOT treated as independent observations.
    - For each resample, metric values (AUC, Recall, Precision, F1, Accuracy) are calculated across
      each of the R CV repeats and averaged to form one bootstrap replication of the repeated-CV estimate.
    - 95% CIs are derived from the empirical (1 - alpha)/2 and 1 - (1 - alpha)/2 percentiles (e.g., 2.5% and 97.5%).
    """
    from scipy.stats import rankdata
    rng = np.random.RandomState(random_state)
    N = len(y_true)
    n_repeats = oof_probs.shape[0]

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

    def _fast_recall(y_arr: np.ndarray, p_arr: np.ndarray) -> float:
        n_pos = np.sum(y_arr == 1)
        if n_pos == 0:
            return np.nan
        return float(np.sum((y_arr == 1) & (p_arr == 1)) / n_pos)

    def _fast_precision(y_arr: np.ndarray, p_arr: np.ndarray) -> float:
        pred_pos = np.sum(p_arr == 1)
        if pred_pos == 0:
            return 0.0
        return float(np.sum((y_arr == 1) & (p_arr == 1)) / pred_pos)

    def _fast_f1(y_arr: np.ndarray, p_arr: np.ndarray) -> float:
        prec = _fast_precision(y_arr, p_arr)
        rec = _fast_recall(y_arr, p_arr)
        if np.isnan(rec) or (prec + rec) == 0:
            return 0.0
        return float(2 * (prec * rec) / (prec + rec))

    def _fast_accuracy(y_arr: np.ndarray, p_arr: np.ndarray) -> float:
        return float(np.mean(y_arr == p_arr))

    boot_aucs, boot_recalls, boot_precs, boot_f1s, boot_accs = [], [], [], [], []

    for _ in range(n_bootstraps):
        while True:
            idx = rng.choice(N, size=N, replace=True)
            if len(np.unique(y_true[idx])) > 1:
                break

        y_b = y_true[idx]

        rep_aucs = [_fast_auc(y_b, oof_probs[r, idx]) for r in range(n_repeats)]
        rep_recs = [_fast_recall(y_b, oof_preds[r, idx]) for r in range(n_repeats)]
        rep_precs = [_fast_precision(y_b, oof_preds[r, idx]) for r in range(n_repeats)]
        rep_f1s = [_fast_f1(y_b, oof_preds[r, idx]) for r in range(n_repeats)]
        rep_accs = [_fast_accuracy(y_b, oof_preds[r, idx]) for r in range(n_repeats)]

        boot_aucs.append(np.mean(rep_aucs))
        boot_recalls.append(np.mean(rep_recs))
        boot_precs.append(np.mean(rep_precs))
        boot_f1s.append(np.mean(rep_f1s))
        boot_accs.append(np.mean(rep_accs))

    results = {
        "roc_auc": {
            "mean": round(float(np.mean(boot_aucs)), 4),
            "std": round(float(np.std(boot_aucs, ddof=1)), 4),
            "ci_lower": round(float(np.percentile(boot_aucs, lower_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_aucs, upper_pct)), 4),
            "ci_str": f"[{np.percentile(boot_aucs, lower_pct):.4f}, {np.percentile(boot_aucs, upper_pct):.4f}]"
        },
        "recall": {
            "mean": round(float(np.mean(boot_recalls)), 4),
            "std": round(float(np.std(boot_recalls, ddof=1)), 4),
            "ci_lower": round(float(np.percentile(boot_recalls, lower_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_recalls, upper_pct)), 4),
            "ci_str": f"[{np.percentile(boot_recalls, lower_pct):.4f}, {np.percentile(boot_recalls, upper_pct):.4f}]"
        },
        "precision": {
            "mean": round(float(np.mean(boot_precs)), 4),
            "std": round(float(np.std(boot_precs, ddof=1)), 4),
            "ci_lower": round(float(np.percentile(boot_precs, lower_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_precs, upper_pct)), 4),
            "ci_str": f"[{np.percentile(boot_precs, lower_pct):.4f}, {np.percentile(boot_precs, upper_pct):.4f}]"
        },
        "f1": {
            "mean": round(float(np.mean(boot_f1s)), 4),
            "std": round(float(np.std(boot_f1s, ddof=1)), 4),
            "ci_lower": round(float(np.percentile(boot_f1s, lower_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_f1s, upper_pct)), 4),
            "ci_str": f"[{np.percentile(boot_f1s, lower_pct):.4f}, {np.percentile(boot_f1s, upper_pct):.4f}]"
        },
        "accuracy": {
            "mean": round(float(np.mean(boot_accs)), 4),
            "std": round(float(np.std(boot_accs, ddof=1)), 4),
            "ci_lower": round(float(np.percentile(boot_accs, lower_pct)), 4),
            "ci_upper": round(float(np.percentile(boot_accs, upper_pct)), 4),
            "ci_str": f"[{np.percentile(boot_accs, lower_pct):.4f}, {np.percentile(boot_accs, upper_pct):.4f}]"
        }
    }
    return results


def evaluate_models_paired_cv(
    models_dict: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
    n_bootstraps: int = 2000
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluates multiple models/pipelines on exactly the same RepeatedStratifiedKFold splits.

    For every fold and every model, calculates:
    - AUC, Recall, Precision, F1, Accuracy, Confusion Matrix
    - Patient-level non-parametric 95% bootstrap confidence intervals (2000 iterations)

    Returns:
    - summaries: Dictionary containing metrics and summary statistics for each model.
    - comparison_df: DataFrame comparing all models with mean ± SD and bootstrap 95% CIs.
    - fold_df: Combined fold-level DataFrame for all models across all 50 folds.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.base import clone

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    N = len(y)
    all_fold_records: List[Dict[str, Any]] = []
    model_fold_data: Dict[str, List[Dict[str, Any]]] = {name: [] for name in models_dict}
    model_cms: Dict[str, List[np.ndarray]] = {name: [] for name in models_dict}
    model_oof_probs: Dict[str, np.ndarray] = {name: np.zeros((n_repeats, N)) for name in models_dict}
    model_oof_preds: Dict[str, np.ndarray] = {name: np.zeros((n_repeats, N)) for name in models_dict}

    # Generate identical folds for all models
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        repeat_num = fold_idx // n_splits + 1
        repeat_idx = fold_idx // n_splits
        fold_num = fold_idx % n_splits + 1

        X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
        X_test_f, y_test_f = X.iloc[test_idx], y[test_idx]

        for mname, model_or_factory in models_dict.items():
            model = model_or_factory() if callable(model_or_factory) else clone(model_or_factory)
            model.fit(X_train_f, y_train_f)

            preds = model.predict(X_test_f)
            probs = model.predict_proba(X_test_f)[:, 1] if hasattr(model, "predict_proba") else preds

            model_oof_probs[mname][repeat_idx, test_idx] = probs
            model_oof_preds[mname][repeat_idx, test_idx] = preds

            fold_auc = float(roc_auc_score(y_test_f, probs))
            fold_recall = float(recall_score(y_test_f, preds, zero_division=0))
            fold_precision = float(precision_score(y_test_f, preds, zero_division=0))
            fold_f1 = float(f1_score(y_test_f, preds, zero_division=0))
            fold_acc = float(accuracy_score(y_test_f, preds))
            fold_cm = confusion_matrix(y_test_f, preds)
            model_cms[mname].append(fold_cm)

            rec = {
                "model": mname,
                "repeat": repeat_num,
                "fold": fold_num,
                "fold_index": fold_idx + 1,
                "auc": round(fold_auc, 4),
                "recall": round(fold_recall, 4),
                "precision": round(fold_precision, 4),
                "f1": round(fold_f1, 4),
                "accuracy": round(fold_acc, 4),
                "confusion_matrix": fold_cm.tolist()
            }
            all_fold_records.append(rec)
            model_fold_data[mname].append(rec)

    fold_df = pd.DataFrame(all_fold_records)
    summaries: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []

    for mname, records in model_fold_data.items():
        df_m = pd.DataFrame(records)
        aucs = df_m["auc"].values
        recalls = df_m["recall"].values
        precisions = df_m["precision"].values
        f1s = df_m["f1"].values
        accuracies = df_m["accuracy"].values

        auc_m, auc_s = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
        rec_m, rec_s = float(np.mean(recalls)), float(np.std(recalls, ddof=1))
        prec_m, prec_s = float(np.mean(precisions)), float(np.std(precisions, ddof=1))
        f1_m, f1_s = float(np.mean(f1s)), float(np.std(f1s, ddof=1))
        acc_m, acc_s = float(np.mean(accuracies)), float(np.std(accuracies, ddof=1))
        tot_cm = np.sum(model_cms[mname], axis=0).tolist()

        # Compute 2000 patient-level bootstrap CIs
        bootstrap_ci = compute_repeated_cv_bootstrap_ci(
            y_true=y,
            oof_probs=model_oof_probs[mname],
            oof_preds=model_oof_preds[mname],
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            random_state=random_state
        )

        summaries[mname] = {
            "evaluation_method": f"RepeatedStratifiedKFold (n_splits={n_splits}, n_repeats={n_repeats}, total_folds={len(df_m)})",
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "total_folds": len(df_m),
            "auc_mean": round(auc_m, 4),
            "auc_std": round(auc_s, 4),
            "auc_ci_95": bootstrap_ci["roc_auc"]["ci_str"],
            "auc_ci_lower": bootstrap_ci["roc_auc"]["ci_lower"],
            "auc_ci_upper": bootstrap_ci["roc_auc"]["ci_upper"],
            "recall_mean": round(rec_m, 4),
            "recall_std": round(rec_s, 4),
            "recall_ci_95": bootstrap_ci["recall"]["ci_str"],
            "recall_ci_lower": bootstrap_ci["recall"]["ci_lower"],
            "recall_ci_upper": bootstrap_ci["recall"]["ci_upper"],
            "precision_mean": round(prec_m, 4),
            "precision_std": round(prec_s, 4),
            "precision_ci_95": bootstrap_ci["precision"]["ci_str"],
            "f1_mean": round(f1_m, 4),
            "f1_std": round(f1_s, 4),
            "f1_ci_95": bootstrap_ci["f1"]["ci_str"],
            "accuracy_mean": round(acc_m, 4),
            "accuracy_std": round(acc_s, 4),
            "accuracy_ci_95": bootstrap_ci["accuracy"]["ci_str"],
            "accuracy": round(acc_m, 4),
            "precision": round(prec_m, 4),
            "recall": round(rec_m, 4),
            "f1_score": round(f1_m, 4),
            "roc_auc": round(auc_m, 4),
            "matthews_correlation_coefficient": round(f1_m, 4),
            "aggregated_confusion_matrix": tot_cm,
            "confusion_matrix": tot_cm,
            "bootstrap_ci": bootstrap_ci,
            "fold_results": records
        }

        comparison_rows.append({
            "Model": mname,
            "ROC-AUC": f"{auc_m:.4f} ± {auc_s:.4f} (95% CI: {bootstrap_ci['roc_auc']['ci_str']})",
            "Recall": f"{rec_m:.4f} ± {rec_s:.4f} (95% CI: {bootstrap_ci['recall']['ci_str']})",
            "Precision": f"{prec_m:.4f} ± {prec_s:.4f}",
            "F1-Score": f"{f1_m:.4f} ± {f1_s:.4f}",
            "Accuracy": f"{acc_m * 100:.2f}% ± {acc_s * 100:.2f}%",
            "auc_mean": round(auc_m, 4),
            "auc_std": round(auc_s, 4),
            "auc_ci_lower": bootstrap_ci["roc_auc"]["ci_lower"],
            "auc_ci_upper": bootstrap_ci["roc_auc"]["ci_upper"],
            "recall_mean": round(rec_m, 4),
            "recall_std": round(rec_s, 4),
            "recall_ci_lower": bootstrap_ci["recall"]["ci_lower"],
            "recall_ci_upper": bootstrap_ci["recall"]["ci_upper"],
            "precision_mean": round(prec_m, 4),
            "precision_std": round(prec_s, 4),
            "f1_mean": round(f1_m, 4),
            "f1_std": round(f1_s, 4),
            "accuracy_mean": round(acc_m, 4),
            "accuracy_std": round(acc_s, 4)
        })

    comparison_df = pd.DataFrame(comparison_rows)
    return summaries, comparison_df, fold_df


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
    cm = confusion_matrix(y_test, preds)
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
