"""Script to generate Chapter 6 Deep Learning figures:
- fig_6_1_resnet18_grouped_confusion_matrix.png
- fig_6_2_per_class_recall_leaky_vs_grouped.png
- fig_6_4_ct_models_benchmark_comparison.png
- fig_6_5_yolo_dinov3_qknn_confusion_matrices.png

(Note: fig_6_3 is marked PENDING per prompt instruction because dl/outputs/charts/roc_curve.png
was generated on the historical leaky model and no test probability vectors exist for ResNet18).
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Georgia', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("docs/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']


def generate_fig_6_1():
    """Figure 6.1: ResNet18 confusion matrix on the GROUPED test set (1,869 slices)."""
    with open("dl/models/ct/resnet18/metrics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    cm = np.array(meta["held_out_test_metrics"]["slice_confusion_matrix"])
    
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Plot heatmap
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"fontsize": 11, "fontweight": "bold", "family": "serif"},
        linewidths=1.0, linecolor="white", cbar_kws={"shrink": 0.8}, ax=ax
    )
    
    ax.set_title("Figure 6.1: ResNet18 Confusion Matrix on Audited Grouped Test Set\n(1,869 Untouched CT Slices, Zero Cross-Split Leakage)",
                 fontsize=11.5, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel("Predicted Pathology Class", fontsize=10.5, fontweight='bold', labelpad=10)
    ax.set_ylabel("Ground Truth Pathology Class", fontsize=10.5, fontweight='bold', labelpad=10)
    
    # Subtitle note
    acc = meta["held_out_test_metrics"]["slice_accuracy"] * 100
    f1 = meta["held_out_test_metrics"]["slice_f1_macro"]
    st_rec = meta["held_out_test_metrics"]["slice_per_class"]["Stone"]["recall"] * 100
    fig.text(0.5, 0.01,
             f"Overall Slice Accuracy: {acc:.2f}% | Macro F1: {f1:.4f} | Stone Recall: {st_rec:.2f}%\n"
             "Source: dl/models/ct/resnet18/metrics.json (Evaluated on Epoch 3 Grouped Checkpoint)",
             ha='center', fontsize=8.5, style='italic', color='#334155')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_file = OUTPUT_DIR / "fig_6_1_resnet18_grouped_confusion_matrix.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_6_2():
    """Figure 6.2: Per-class recall, leaky vs grouped."""
    # Historical leaky baseline from dl/outputs/reports/classification_report.txt and audit report
    leaky_recalls = [100.0, 100.0, 97.12, 99.71]
    
    # Audited grouped held-out test split from dl/models/ct/resnet18/metrics.json
    grouped_recalls = [70.74, 98.69, 65.70, 100.0]
    
    # Cluster consensus recall (mean probability voting)
    cluster_recalls = [97.56, 100.0, 80.95, 100.0]

    x = np.arange(len(CLASS_NAMES))
    width = 0.28
    
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
    fig.patch.set_facecolor('white')
    
    b1 = ax.bar(x - width, leaky_recalls, width, label='Historical Leaky Baseline (Discredited Slice Shuffle)',
                color='#f87171', edgecolor='#991b1b', lw=1.0)
    b2 = ax.bar(x, grouped_recalls, width, label='Audited Grouped Test Set (Slice-Level Evaluation)',
                color='#3b82f6', edgecolor='#1e40af', lw=1.0)
    b3 = ax.bar(x + width, cluster_recalls, width, label='Audited Grouped Test Set (Cluster Consensus Voting)',
                color='#10b981', edgecolor='#065f46', lw=1.0)

    # Value labels
    for bars, col in [(b1, '#991b1b'), (b2, '#1e40af'), (b3, '#065f46')]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 1.2,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=col)

    ax.set_title("Figure 6.2: Per-Class Diagnostic Sensitivity (Recall) — Historical Leaky vs. Audited Grouped Evaluation",
                 fontsize=11.5, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel("Kidney Pathology Class", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Sensitivity / Recall (%)", fontsize=10.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9)

    fig.text(0.5, 0.01,
             "*Note: Eliminating 227 duplicate pairs and 703 cross-split clusters reveals true slice Stone recall drops from 97.12% to 65.70%,\n"
             "recovering to 80.95% under multi-slice cluster consensus voting. Tumor sensitivity remains 100.0% in grouped test partition.",
             ha='center', fontsize=8, style='italic', color='#475569')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_file = OUTPUT_DIR / "fig_6_2_per_class_recall_leaky_vs_grouped.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_6_4():
    """Figure 6.4: ResNet18, YOLO26, DINOv3, QKNN comparison on the same 1,869-slice grouped test set."""
    models = ['ResNet18\n(CNN Baseline)', 'YOLO26\n(SOTA Classifier)', 'DINOv3 + MLP\n(Self-Supervised)', 'QKNN\n(Quantum Prototype)']
    
    # Metrics confirmed from predictions.csv and metrics.json on 1,869 slices
    accuracy = [86.94, 92.19, 87.43, 41.95]
    macro_f1 = [82.51, 90.47, 82.81, 35.89]
    stone_recall = [65.70, 94.20, 61.84, 25.60]
    stone_precision = [45.95, 69.40, 51.82, 9.55]

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    fig.patch.set_facecolor('white')

    x = np.arange(len(models))
    width = 0.20

    b1 = ax.bar(x - 1.5 * width, accuracy, width, label='Overall Accuracy (%)', color='#2563eb', edgecolor='#1e3a8a', lw=0.8)
    b2 = ax.bar(x - 0.5 * width, macro_f1, width, label='Macro F1-Score (%)', color='#0d9488', edgecolor='#115e59', lw=0.8)
    b3 = ax.bar(x + 0.5 * width, stone_recall, width, label='Stone Recall / Sensitivity (%)', color='#f59e0b', edgecolor='#b45309', lw=0.8)
    b4 = ax.bar(x + 1.5 * width, stone_precision, width, label='Stone Precision (%)', color='#e11d48', edgecolor='#9f1239', lw=0.8)

    for bars, col in [(b1, '#1e3a8a'), (b2, '#115e59'), (b3, '#b45309'), (b4, '#9f1239')]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 1.2,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold', color=col)

    ax.set_title("Figure 6.4: Multi-Model Benchmark on Identical 1,869-Slice Grouped Held-Out Test Set",
                 fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xlabel("Evaluated Deep Learning Architecture", fontsize=10.5, fontweight='bold', labelpad=8)
    ax.set_ylabel("Performance Metric (%)", fontsize=10.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)

    fig.text(0.5, 0.01,
             "All models benchmarked strictly on the audited grouped test set (1,869 slices, zero cross-split duplicate leakage).\n"
             "Sources: dl/models/ct/resnet18/metrics.json, dl/models/ct/yolo26/predictions.csv, dinov3/predictions.csv, qknn/predictions.csv",
             ha='center', fontsize=8, style='italic', color='#475569')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_file = OUTPUT_DIR / "fig_6_4_ct_models_benchmark_comparison.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_6_5():
    """Figure 6.5: Confusion matrices for YOLO26, DINOv3+MLP, QKNN from their predictions.csv."""
    models_info = [
        ("YOLO26", "dl/models/ct/yolo26/predictions.csv", "Greens", "Acc: 92.19% | Macro F1: 0.9047\nStone Recall: 94.20%"),
        ("DINOv3 + MLP", "dl/models/ct/dinov3/predictions.csv", "Blues", "Acc: 87.43% | Macro F1: 0.8281\nStone Recall: 61.84%"),
        ("QKNN", "dl/models/ct/qknn/predictions.csv", "Purples", "Acc: 41.95% | Macro F1: 0.3589\nStone Recall: 25.60%")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=300)
    fig.patch.set_facecolor('white')

    for ax, (mname, ppath, cmap, summary_txt) in zip(axes, models_info):
        df = pd.read_csv(ppath)
        cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=[0, 1, 2, 3])
        
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"fontsize": 10, "fontweight": "bold", "family": "serif"},
            linewidths=1.0, linecolor="white", cbar=False, ax=ax
        )
        
        ax.set_title(f"Model: {mname}\n{summary_txt}", fontsize=10.5, fontweight='bold', pad=10, color='#0f172a')
        ax.set_xlabel("Predicted Class", fontsize=9.5, fontweight='bold')
        ax.set_ylabel("True Class", fontsize=9.5, fontweight='bold')

    fig.suptitle("Figure 6.5: Test Set Confusion Matrices for Alternative CT Classifiers (1,869 Grouped Slices)",
                 fontsize=13, fontweight='bold', y=0.98, color='#0f172a')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_file = OUTPUT_DIR / "fig_6_5_yolo_dinov3_qknn_confusion_matrices.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


if __name__ == "__main__":
    generate_fig_6_1()
    generate_fig_6_2()
    generate_fig_6_4()
    generate_fig_6_5()
