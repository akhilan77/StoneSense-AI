"""Script to generate Chapter 3 system architecture and dual-stream figures.
Outputs:
- docs/report_figures/fig_3_1_system_architecture.png
- docs/report_figures/fig_3_2_independent_streams.png
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Georgia', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("docs/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, title, items, color_bg, color_border, title_color="#111827", body_color="#374151"):
    """Helper to draw styled architectural component box."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=color_bg,
        edgecolor=color_border,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)
    
    # Title
    ax.text(
        x + w / 2, y + h - 0.18, title,
        ha="center", va="top",
        fontsize=11, fontweight="bold",
        color=title_color, zorder=3
    )
    
    # Divider line
    ax.plot([x + 0.1, x + w - 0.1], [y + h - 0.32, y + h - 0.32], color=color_border, lw=1.0, zorder=3)
    
    # Bullet points / descriptions
    y_pos = y + h - 0.48
    for item in items:
        ax.text(
            x + 0.15, y_pos, f"•  {item}",
            ha="left", va="top",
            fontsize=9, color=body_color, zorder=3
        )
        y_pos -= 0.22


def generate_fig_3_1():
    """Generates Figure 3.1: StoneSense-AI End-to-End System Architecture."""
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Overall Title & Subtitle
    ax.text(6.0, 8.2, "StoneSense-AI: Comprehensive System Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(6.0, 7.85, "Decoupled Clinical Workflows, FastAPI Telemetry, Federated Coordinator & Multi-Model Inference",
            ha="center", va="center", fontsize=10, style="italic", color="#475569")

    # Layer 1: Client Layer (Top)
    l1_bg = patches.Rectangle((0.5, 5.8), 11.0, 1.7, facecolor="#f8fafc", edgecolor="#cbd5e1", linestyle="--", lw=1.0, zorder=1)
    ax.add_patch(l1_bg)
    ax.text(0.7, 7.3, "PRESENTATION LAYER (React 18 + Vite + Tailwind CSS)", fontsize=9, fontweight="bold", color="#64748b")

    draw_box(ax, 0.8, 5.95, 3.2, 1.35, "Hospital Clinical Portal",
             ["Urine Risk Evaluation", "CT Stone Classification", "Local Telemetry History"],
             "#f0fdf4", "#86efac", "#166534")

    draw_box(ax, 4.4, 5.95, 3.2, 1.35, "Federated Learning Console",
             ["Live Round Monitor", "Flower Node Status (HOSP 1-3)", "13 WebSocket Event Feeds"],
             "#eff6ff", "#93c5fd", "#1e40af")

    draw_box(ax, 8.0, 5.95, 3.2, 1.35, "Developer & Safety Dashboard",
             ["Candidate Model Registry", "Automated Gate Decisions", "Model Promotion / Rollback"],
             "#faf5ff", "#d8b4fe", "#6b21a8")

    # Connectors Layer 1 to Layer 2 (REST / WebSocket)
    for cx in [2.4, 6.0, 9.6]:
        ax.annotate("", xy=(cx, 5.3), xytext=(cx, 5.95),
                    arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.5))
    ax.text(6.0, 5.55, "REST API (JSON / Multipart)  &  WebSocket Telemetry (/ws/federated)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#334155",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", lw=0.8))

    # Layer 2: API & Application Core (FastAPI)
    l2_bg = patches.Rectangle((0.5, 3.0), 11.0, 2.3, facecolor="#f8fafc", edgecolor="#cbd5e1", linestyle="--", lw=1.0, zorder=1)
    ax.add_patch(l2_bg)
    ax.text(0.7, 5.1, "APPLICATION & ORCHESTRATION LAYER (FastAPI + Uvicorn)", fontsize=9, fontweight="bold", color="#64748b")

    draw_box(ax, 0.8, 3.15, 3.2, 1.8, "API Routers & Auth",
             ["JWT Security & Scoping", "Predict Router (/predict)", "Hospital & Patient API", "Audit Logging Engine"],
             "#ffffff", "#cbd5e1", "#0f172a")

    draw_box(ax, 4.4, 3.15, 3.2, 1.8, "Federated Coordinator",
             ["Single-Round Concurrency Lock", "FedAvg Aggregator", "13-Event Broadcast Engine", "WebSocket Connection Manager"],
             "#ffffff", "#cbd5e1", "#0f172a")

    draw_box(ax, 8.0, 3.15, 3.2, 1.8, "Deployment Safety Gate",
             ["Acc >= 0.95, F1 >= 0.94", "Stone Recall >= 0.96", "Tumor Recall >= 0.98", "Regression Delta <= 0.02"],
             "#fff1f2", "#fecdd3", "#9f1239")

    # Connectors Layer 2 to Layer 3
    for cx in [2.4, 6.0, 9.6]:
        ax.annotate("", xy=(cx, 2.4), xytext=(cx, 3.15),
                    arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.5))

    # Layer 3: Persistence & Model Registry (Bottom)
    l3_bg = patches.Rectangle((0.5, 0.4), 11.0, 2.0, facecolor="#f8fafc", edgecolor="#cbd5e1", linestyle="--", lw=1.0, zorder=1)
    ax.add_patch(l3_bg)
    ax.text(0.7, 2.2, "DATA, MODEL REGISTRY & ISOLATION LAYER", fontsize=9, fontweight="bold", color="#64748b")

    draw_box(ax, 0.8, 0.55, 3.2, 1.5, "Relational DB (stonesense.db)",
             ["12 SQLAlchemy Tables", "hospitals, users, predictions", "federated_rounds, audit_logs"],
             "#f0fdfa", "#99f6e4", "#115e59")

    draw_box(ax, 4.4, 0.55, 3.2, 1.5, "ML & DL Model Registry",
             ["ResNet18 (ImageNet fine-tuned)", "YOLO26, DINOv3, QKNN", "candidate_risk_model.pkl (LR)"],
             "#fdf4ff", "#f5d0fe", "#86198f")

    draw_box(ax, 8.0, 0.55, 3.2, 1.5, "Isolated Hospital Nodes",
             ["HOSP-001 (Apollo Care)", "HOSP-002 (Manipal Inst.)", "HOSP-003 (AIIMS Labs)", "Zero-Raw-CT Privacy Barrier"],
             "#fefce8", "#fef08a", "#854d0e")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig_3_1_system_architecture.png"
    plt.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {output_path} (width >= 1800 px)")


def generate_fig_3_2():
    """Generates Figure 3.2: Dual Independent Diagnostic Streams (No Multimodal Fusion)."""
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Title & Subtitle
    ax.text(6.0, 7.2, "StoneSense-AI: Dual Independent Diagnostic Streams",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")
    ax.text(6.0, 6.85, "Strictly Decoupled Biochemical and Radiological Processing (Zero Multimodal Feature Fusion)",
            ha="center", va="center", fontsize=10, style="italic", color="#475569")

    # Stream 1: Urine Biochemical Analysis (Top Stream)
    s1_bg = patches.Rectangle((0.5, 3.65), 11.0, 2.9, facecolor="#f0fdf4", edgecolor="#bbf7d0", linewidth=1.5, zorder=1)
    ax.add_patch(s1_bg)
    ax.text(0.7, 6.3, "STREAM 1: URINE BIOCHEMICAL RISK PIPELINE (Model B)", fontsize=10, fontweight="bold", color="#166534")

    draw_box(ax, 0.8, 3.85, 2.2, 2.25, "Clinical Input",
             ["6 Urine Biomarkers", "Gravity, pH, Osmo,", "Conductivity,", "Urea, Calcium", "(kidneyData.csv)"],
             "#ffffff", "#86efac", "#166534")

    draw_box(ax, 3.5, 3.85, 2.2, 2.25, "Preprocessing",
             ["Elementwise Ratios", "cond/osmo, urea/calc", "StandardScaler", "No Imputation / SMOTE", "(79 Observations)"],
             "#ffffff", "#86efac", "#166534")

    draw_box(ax, 6.2, 3.85, 2.3, 2.25, "Inference Engine",
             ["Logistic Regression", "1-SE Rule Winner", "Nested 50-Fold CV", "ROC-AUC: 0.8387", "Nadeau-Bengio SE"],
             "#ffffff", "#86efac", "#166534")

    draw_box(ax, 9.0, 3.85, 2.3, 2.25, "Clinical Output",
             ["Binary Risk Score", "(Low vs High Risk)", "Local SHAP Waterfall", "Feature Attributions", "No Patient PII"],
             "#ffffff", "#86efac", "#166534")

    # Arrows for Stream 1
    for ax_start, ax_end in [(3.0, 3.5), (5.7, 6.2), (8.5, 9.0)]:
        ax.annotate("", xy=(ax_end, 5.0), xytext=(ax_start, 5.0),
                    arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.0))

    # Separation Barrier Banner
    banner = patches.Rectangle((0.5, 3.3), 11.0, 0.25, facecolor="#fee2e2", edgecolor="#fca5a5", linewidth=1.0, zorder=2)
    ax.add_patch(banner)
    ax.text(6.0, 3.425, "STRICT ARCHITECTURAL SEPARATION: NO MULTIMODAL CONCATENATION OR SHARED LATENT EMBEDDING",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#991b1b")

    # Stream 2: Radiological CT Classification (Bottom Stream)
    s2_bg = patches.Rectangle((0.5, 0.3), 11.0, 2.9, facecolor="#eff6ff", edgecolor="#bfdbfe", linewidth=1.5, zorder=1)
    ax.add_patch(s2_bg)
    ax.text(0.7, 2.95, "STREAM 2: RADIOLOGICAL CT STONE DETECTION PIPELINE (Model A)", fontsize=10, fontweight="bold", color="#1e40af")

    draw_box(ax, 0.8, 0.5, 2.2, 2.25, "Radiological Input",
             ["Axial CT Scans", "512x512 JPEG Slices", "12,446 Total Slices", "Flat Directory", "(No DICOM UIDs)"],
             "#ffffff", "#93c5fd", "#1e40af")

    draw_box(ax, 3.5, 0.5, 2.2, 2.25, "Cluster Splitting",
             ["64-bit DCT pHash <= 2", "979 Graph Clusters", "Zero Duplicate Leak", "70/15/15 Grouped", "(Train/Val/Test)"],
             "#ffffff", "#93c5fd", "#1e40af")

    draw_box(ax, 6.2, 0.5, 2.3, 2.25, "Deep Learning Core",
             ["Torchvision ResNet18", "Pretrained ImageNet", "Fine-tuned Layer4+FC", "YOLO26 / DINOv3 / QKNN", "Grouped Test: 86.94%"],
             "#ffffff", "#93c5fd", "#1e40af")

    draw_box(ax, 9.0, 0.5, 2.3, 2.25, "Diagnostic Output",
             ["4-Class Prediction", "(Normal, Cyst, Stone, Tumor)", "Grad-CAM Attention Map", "Mean Cluster Voting", "(96.48% Cluster Acc)"],
             "#ffffff", "#93c5fd", "#1e40af")

    # Arrows for Stream 2
    for ax_start, ax_end in [(3.0, 3.5), (5.7, 6.2), (8.5, 9.0)]:
        ax.annotate("", xy=(ax_end, 1.65), xytext=(ax_start, 1.65),
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.0))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig_3_2_independent_streams.png"
    plt.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {output_path} (width >= 1800 px)")


if __name__ == "__main__":
    generate_fig_3_1()
    generate_fig_3_2()
