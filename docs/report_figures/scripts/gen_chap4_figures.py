"""Script to generate Chapter 4 figures: CT dataset distribution, real CT slice grid,
pHash grouping workflow, and urine biomarker correlation heatmap.

Outputs:
- docs/report_figures/fig_4_1_ct_class_and_split_distribution.png
- docs/report_figures/fig_4_2_ct_sample_slices_grid.png
- docs/report_figures/fig_4_3_phash_grouping_workflow.png
- docs/report_figures/fig_4_4_urine_biomarker_correlation_heatmap.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Georgia', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("docs/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_fig_4_1():
    """Figure 4.1: CT class counts & grouped train/val/test split from grouped_split_manifest.csv."""
    manifest_path = Path("dl/outputs/reports/grouped_split_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    # Cross tabulation of class and destination_split
    ct = pd.crosstab(df['class'], df['destination_split'])
    # Order splits
    splits = ['train', 'validation', 'test']
    classes = ['Normal', 'Cyst', 'Tumor', 'Stone']
    
    data = np.array([[ct.loc[c, s] for s in splits] for c in classes])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300, gridspec_kw={'width_ratios': [1.3, 1]})
    fig.patch.set_facecolor('white')
    
    # Left subplot: Grouped bar chart of slice counts per class across splits
    x = np.arange(len(classes))
    width = 0.25
    colors = ['#1e3a8a', '#0284c7', '#38bdf8']
    split_labels = ['Train (70%)', 'Validation (15%)', 'Test (15%)']
    
    for i, s in enumerate(splits):
        bars = ax1.bar(x + (i - 1) * width, data[:, i], width, label=split_labels[i], color=colors[i], edgecolor='#1e293b', lw=0.8)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 45,
                     f'{int(height):,}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1e293b')
            
    ax1.set_title("CT Slice Distribution by Class and Grouped Partition", fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Kidney Pathology Class", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Slice Count", fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 4200)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)
    
    # Right subplot: Total split summary & cluster counts
    split_totals = df['destination_split'].value_counts()[splits]
    # Cluster counts per split
    cluster_counts = df.groupby('destination_split')['near_duplicate_group_id'].nunique()[splits]
    
    y = np.arange(len(splits))
    bar_height = 0.35
    
    ax2.barh(y + bar_height/2, split_totals.values, bar_height, label='Total Slices', color='#2563eb', edgecolor='#1e293b', lw=0.8)
    for idx, val in enumerate(split_totals.values):
        pct = (val / len(df)) * 100
        ax2.text(val + 150, idx + bar_height/2, f"{val:,} ({pct:.1f}%)", va='center', fontsize=8.5, fontweight='bold', color='#1e293b')
        
    ax2.barh(y - bar_height/2, cluster_counts.values * 10, bar_height, label='Clusters (x10 scale)', color='#10b981', edgecolor='#1e293b', lw=0.8)
    for idx, val in enumerate(cluster_counts.values):
        ax2.text(val * 10 + 150, idx - bar_height/2, f"{val} clusters", va='center', fontsize=8.5, fontweight='bold', color='#047857')

    ax2.set_title("Grouped Partition Composition & Cluster Allocation", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("Count", fontsize=10, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(['Train Split', 'Validation Split', 'Test Split'], fontsize=10, fontweight='bold')
    ax2.set_xlim(0, 10500)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5, loc='lower right')
    
    # Header note
    fig.suptitle("Figure 4.1: StoneSense-AI Grouped CT Dataset Stratification (12,446 Slices across 979 Clusters)",
                 fontsize=13, fontweight='bold', y=0.98, color='#0f172a')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_file = OUTPUT_DIR / "fig_4_1_ct_class_and_split_distribution.png"
    plt.savefig(out_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_4_2():
    """Figure 4.2: 1x4 grid of real CT slices (Normal, Cyst, Tumor, Stone) from dl/processed_grouped/train/."""
    classes = [
        ("Normal", "Normal- (1).jpg", "Normal Renal Parenchyma\n(Homogeneous cortex, intact corticomedullary boundary)"),
        ("Cyst", "Cyst- (100).jpg", "Renal Cortical Cyst\n(Well-circumscribed fluid-attenuating lesion, thin wall)"),
        ("Tumor", "Tumor- (10).jpg", "Solid Renal Neoplasm\n(Heterogeneous mass disrupting renal contour)"),
        ("Stone", "Stone- (1).jpg", "Renal Calculus (Stone)\n(Hyperdense radio-opaque calculus in pelvicalyceal system)")
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), dpi=300)
    fig.patch.set_facecolor('white')
    
    for ax, (cname, fname, desc) in zip(axes, classes):
        img_path = Path(f"dl/processed_grouped/train/{cname}/{fname}")
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"Class: {cname}\nFile: {fname}", fontsize=11, fontweight='bold', pad=8, color='#0f172a')
        ax.axis('off')
        
        # Add descriptive text below image
        ax.text(0.5, -0.08, desc, transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, style='italic', color='#334155')
        
    fig.suptitle("Figure 4.2: Representative 512×512 Axial CT Slices from Grouped Training Cohort",
                 fontsize=13, fontweight='bold', y=0.98, color='#0f172a')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    out_file = OUTPUT_DIR / "fig_4_2_ct_sample_slices_grid.png"
    plt.savefig(out_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_4_3():
    """Figure 4.3: pHash grouping workflow from real implementation (dl/preprocessing)."""
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Title
    ax.text(6.5, 7.1, "Perceptual Hash (pHash) Grouping & Leakage Elimination Workflow",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(6.5, 6.75, "Transformation of 12,446 Unlabeled JPEG Slices into 979 Atomically Split Perceptual Clusters",
            ha="center", va="center", fontsize=10, style="italic", color="#475569")

    def draw_proc_box(x, y, w, h, step_num, title, items, bg_col, border_col):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=bg_col, edgecolor=border_col, linewidth=1.5, zorder=2
        )
        ax.add_patch(box)
        # Step badge
        badge = patches.Circle((x + 0.28, y + h - 0.25), 0.18, facecolor=border_col, edgecolor="white", zorder=3)
        ax.add_patch(badge)
        ax.text(x + 0.28, y + h - 0.25, str(step_num), ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)
        
        ax.text(x + 0.55, y + h - 0.25, title, ha="left", va="center", fontsize=10.5, fontweight="bold", color="#0f172a", zorder=3)
        ax.plot([x + 0.1, x + w - 0.1], [y + h - 0.45, y + h - 0.45], color=border_col, lw=0.8, zorder=3)
        
        y_cur = y + h - 0.65
        for it in items:
            ax.text(x + 0.15, y_cur, f"• {it}", ha="left", va="top", fontsize=8.5, color="#334155", zorder=3)
            y_cur -= 0.22

    # Step 1: Raw Image Dataset Audit
    draw_proc_box(0.5, 3.8, 3.6, 2.5, 1, "Forensic Dataset Audit",
                  ["12,446 Flat JPEG CT Slices",
                   "Stripped DICOM UIDs & Patient IDs",
                   "MD5 Hashing: 227 duplicate pairs (454 files)",
                   "Slice Shuffle: 43.9% duplicate cross-leakage",
                   "Status: Discredited 99.63% test accuracy"],
                  "#fff1f2", "#fb7185")

    # Step 2: 64-bit DCT Perceptual Hashing
    draw_proc_box(4.7, 3.8, 3.6, 2.5, 2, "Perceptual Hash Extraction",
                  ["Grayscale conversion & 32×32 bicubic resize",
                   "Discrete Cosine Transform (DCT) matrix",
                   "Extract top-left 8×8 low-frequency DCT block",
                   "Exclude DC component at (0,0)",
                   "Median thresholding -> 64-bit binary bitstring",
                   "Quantifies anatomical spatial similarity"],
                  "#eff6ff", "#60a5fa")

    # Step 3: Graph Clustering via Hamming Distance
    draw_proc_box(8.9, 3.8, 3.6, 2.5, 3, "Connected Components Graph",
                  ["Pairwise 64-bit Hamming Distance computation",
                   "Edge condition: Hamming Distance <= 2 bits",
                   "Build adjacency graph: G = (V, E)",
                   "Find connected components (indivisible units)",
                   "Yields 979 disjoint perceptual clusters:",
                   "  829 multi-slice scans + 150 singletons"],
                  "#f5f3ff", "#a78bfa")

    # Arrows between top steps
    ax.annotate("", xy=(4.7, 5.05), xytext=(4.1, 5.05), arrowprops=dict(arrowstyle="->", color="#64748b", lw=2.0))
    ax.annotate("", xy=(8.9, 5.05), xytext=(8.3, 5.05), arrowprops=dict(arrowstyle="->", color="#64748b", lw=2.0))

    # Downward arrow to split phase
    ax.annotate("", xy=(6.5, 3.1), xytext=(6.5, 3.8), arrowprops=dict(arrowstyle="->", color="#64748b", lw=2.0))

    # Step 4: Atomic Cluster-Level Stratified Partitioning (Bottom Box)
    b4 = patches.FancyBboxPatch(
        (0.5, 0.5), 12.0, 2.4, boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#f0fdf4", edgecolor="#4ade80", linewidth=1.5, zorder=2
    )
    ax.add_patch(b4)
    
    badge4 = patches.Circle((0.9, 2.55), 0.18, facecolor="#4ade80", edgecolor="white", zorder=3)
    ax.add_patch(badge4)
    ax.text(0.9, 2.55, "4", ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)
    
    ax.text(1.2, 2.55, "Atomic Cluster-Level Partitioning & Leakage-Free Generalization",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#14532d", zorder=3)
    ax.plot([0.7, 12.3], [2.35, 2.35], color="#4ade80", lw=0.8, zorder=3)

    # 3 sub-columns inside Step 4
    col1 = ["Cluster Stratification Protocol:",
            "• Clusters assigned atomically to splits",
            "• Class-ratio preserving greedy allocation",
            "• 0 exact duplicate pairs cross splits (MD5)",
            "• 0 perceptual clusters cross splits (pHash <= 2)"]
    
    col2 = ["Partition Allocations:",
            "• Train: 8,708 slices (69.97%) | 699 clusters",
            "• Validation: 1,869 slices (15.02%) | 138 clusters",
            "• Held-out Test: 1,869 slices (15.02%) | 142 clusters",
            "• Total: 12,446 slices reconciled across 979 clusters"]

    col3 = ["Audited Empirical Outcomes:",
            "• ResNet18 True Test Accuracy: 86.94%",
            "• ResNet18 True Macro F1: 0.8251",
            "• Cluster Consensus Voting Accuracy: 96.48%",
            "• Cluster Consensus Macro F1: 0.9547"]

    y_t = 2.15
    for c_text, x_pos in [(col1, 0.7), (col2, 4.7), (col3, 8.8)]:
        y_sub = y_t
        for line in c_text:
            is_hdr = ":" in line and not line.startswith("•")
            ax.text(x_pos, y_sub, line, ha="left", va="top",
                    fontsize=8.5, fontweight="bold" if is_hdr else "normal",
                    color="#0f172a" if is_hdr else "#334155", zorder=3)
            y_sub -= 0.22

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig_4_3_phash_grouping_workflow.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_4_4():
    """Figure 4.4: Urine biomarker correlation heatmap from ml/datasets/kidneyData.csv (6 features)."""
    data_path = Path("ml/datasets/kidneyData.csv")
    df = pd.read_csv(data_path)
    
    features = ['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']
    feature_labels = [
        'Specific Gravity\n(gravity)',
        'pH\n(ph)',
        'Osmolality\n(osmo, mOsm)',
        'Conductivity\n(cond, mMho)',
        'Urea\n(urea, mmol/L)',
        'Calcium\n(calc, mmol/L)'
    ]
    
    corr = df[features].corr(method='pearson')
    
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Mask upper triangle for cleaner academic presentation
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0,
        annot=True, fmt=".2f", annot_kws={"fontsize": 10, "fontweight": "bold", "family": "serif"},
        linewidths=1.0, linecolor="white", cbar_kws={"label": "Pearson Correlation Coefficient (r)", "shrink": 0.85},
        xticklabels=feature_labels, yticklabels=feature_labels, ax=ax
    )
    
    ax.set_title("Figure 4.4: Urine Biomarker Inter-Correlation Heatmap (N = 79)",
                 fontsize=12, fontweight='bold', pad=15, color='#0f172a')
    
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig_4_4_urine_biomarker_correlation_heatmap.png"
    plt.savefig(out_file, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Generated {out_file}")


if __name__ == "__main__":
    generate_fig_4_1()
    generate_fig_4_2()
    generate_fig_4_3()
    generate_fig_4_4()
