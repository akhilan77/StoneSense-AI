"""Evaluation helper functions for StoneSense-AI (Model A).

Calculates model evaluation metrics (Accuracy, Precision, Recall, F1)
and saves evaluation charts (confusion matrix, ROC curve, Precision-Recall curve).
"""

from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
)

logger = logging.getLogger("DLEvaluationHelpers")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates model performance on test set and computes comprehensive metrics."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    all_probs_arr = np.array(all_probs)

    # Calculate global metrics
    acc = float(accuracy_score(all_targets_arr, all_preds_arr))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets_arr, all_preds_arr, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets_arr, all_preds_arr, average='weighted', zero_division=0)

    # Calculate per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(all_targets_arr, all_preds_arr, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1_score": float(f1_class[idx]),
            "support": int(support_class[idx])
        }

    cls_report = classification_report(all_targets_arr, all_preds_arr, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets_arr, all_preds_arr).tolist()

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class_metrics": per_class_metrics,
        "classification_report": cls_report,
        "confusion_matrix": cm
    }

    return metrics, all_targets_arr, all_preds_arr, all_probs_arr


def save_evaluation_charts(
    targets: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    charts_dir: Path
) -> None:
    """Generates and saves confusion matrix, ROC curves, and PR curves."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("CT Kidney Classification - Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("True Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(charts_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curves (One-vs-Rest)
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(class_names):
        binary_targets = (targets == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_targets, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC = {roc_auc:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess")
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(charts_dir / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(class_names):
        binary_targets = (targets == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_targets, probs[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{cname} (PR-AUC = {pr_auc:.3f})", lw=2)

    plt.xlabel("Recall", fontweight="bold")
    plt.ylabel("Precision", fontweight="bold")
    plt.title("Precision-Recall Curves", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(charts_dir / "precision_recall_curve.png", dpi=300)
    plt.close()

    logger.info(f"Evaluation charts successfully saved to {charts_dir}")
