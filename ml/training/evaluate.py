"""Evaluation utilities for ML Risk Prediction model.

Computes metrics (Accuracy, F1, MCC, Cohen's Kappa, ROC-AUC) and exports plots.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, List
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
