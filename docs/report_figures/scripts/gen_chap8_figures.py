"""Script to generate Chapter 8 Federated Learning figures:
- fig_8_1_federated_system_topology.png: 3 hospital clients, coordinator, FedAvg, global model
- fig_8_2_federated_round_event_sequence.png: Sequence of 1 round with 13 coordinator events
- fig_8_3_federated_training_convergence.png: Train loss & Validation Acc/F1 from stonesense.db
- fig_8_4_deployment_gate_flowchart.png: Deployment gate flowchart matching deployment_gate.py

Outputs go in docs/report_figures/.
"""

import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Georgia', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("docs/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_fig_8_1():
    """Figure 8.1: Federated Multi-Hospital System Topology."""
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(6.0, 7.15, "StoneSense-AI: Federated Learning Network Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(6.0, 6.8, "Privacy-Preserving Multi-Hospital Collaborative Training (Flower FedAvg & Zero-Raw-CT Isolation)",
            ha="center", va="center", fontsize=10, style="italic", color="#475569")

    # 3 Hospital Client Boxes (Top)
    hospitals = [
        ("Hospital Node 1 (HOSP-001)", "Apollo Renal Care (Chennai)",
         ["Isolated Private CT Partition", "Local Optimizer (Adam lr=1e-3)", "Local Epochs = 1", "Weights-Only Export"],
         0.6, "#f0fdf4", "#86efac", "#166534"),
        ("Hospital Node 2 (HOSP-002)", "Manipal Urology Inst. (Bengaluru)",
         ["Isolated Private CT Partition", "Local Optimizer (Adam lr=1e-3)", "Local Epochs = 1", "Weights-Only Export"],
         4.4, "#eff6ff", "#93c5fd", "#1e40af"),
        ("Hospital Node 3 (HOSP-003)", "AIIMS Nephrology (Delhi)",
         ["Isolated Private CT Partition", "Local Optimizer (Adam lr=1e-3)", "Local Epochs = 1", "Weights-Only Export"],
         8.2, "#faf5ff", "#d8b4fe", "#6b21a8")
    ]

    for title, subtitle, items, x_pos, bg, border, tc in hospitals:
        box = patches.FancyBboxPatch((x_pos, 4.2), 3.2, 2.3, boxstyle="round,pad=0.03,rounding_size=0.06",
                                     facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x_pos + 1.6, 6.25, title, ha="center", va="center", fontsize=10, fontweight="bold", color=tc)
        ax.text(x_pos + 1.6, 6.0, subtitle, ha="center", va="center", fontsize=8.5, style="italic", color="#475569")
        ax.plot([x_pos + 0.1, x_pos + 3.1], [5.85, 5.85], color=border, lw=0.8)
        y_it = 5.65
        for it in items:
            ax.text(x_pos + 0.15, y_it, f"• {it}", fontsize=8.5, color="#1e293b")
            y_it -= 0.22

        # Up/Down Arrows between Hospital and Coordinator
        ax.annotate("", xy=(x_pos + 1.6, 3.4), xytext=(x_pos + 1.6, 4.2),
                    arrowprops=dict(arrowstyle="<->", color=border, lw=2.0))
        ax.text(x_pos + 1.6, 3.8, "NumPy Weights\n(Zero Raw Data)", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#334155",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#cbd5e1", lw=0.6))

    # Central Coordinator Box (Middle)
    coord_box = patches.FancyBboxPatch((0.6, 1.4), 10.8, 1.9, boxstyle="round,pad=0.03,rounding_size=0.06",
                                       facecolor="#f8fafc", edgecolor="#475569", linewidth=1.8, zorder=2)
    ax.add_patch(coord_box)
    ax.text(6.0, 3.05, "Central Federated Coordinator (backend/app/services/federated_coordinator.py)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#0f172a")
    ax.plot([0.8, 11.2], [2.85, 2.85], color="#94a3b8", lw=0.8)

    c_items = [
        ("Concurrency & Orchestration:", ["Single-round concurrency lock (threading.Lock)", "Hospital node selection & validation dataset verification"]),
        ("Aggregation Engine (FedAvg):", ["Sample-weighted parameter averaging across clients", "Evaluation on combined hospital validation partitions"]),
        ("Telemetry & Lifecycle Broadcasting:", ["13 distinct WebSocket lifecycle events to frontend consoles", "Versioned model persistence (resnet18_fed_round_00N.pth)"])
    ]
    for idx, (col_hdr, sub_its) in enumerate(c_items):
        cx = 0.8 + idx * 3.6
        ax.text(cx, 2.65, col_hdr, fontsize=8.5, fontweight="bold", color="#1e3a8a")
        y_sub = 2.45
        for si in sub_its:
            ax.text(cx, y_sub, f"• {si}", fontsize=8, color="#334155")
            y_sub -= 0.2

    # Bottom Telemetry & Gate Row
    ax.annotate("", xy=(3.5, 0.9), xytext=(3.5, 1.4), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))
    ax.annotate("", xy=(8.5, 0.9), xytext=(8.5, 1.4), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))

    db_box = patches.FancyBboxPatch((0.6, 0.2), 5.2, 0.7, boxstyle="round,pad=0.03,rounding_size=0.05",
                                    facecolor="#f0fdfa", edgecolor="#14b8a6", linewidth=1.2, zorder=2)
    ax.add_patch(db_box)
    ax.text(3.2, 0.55, "SQLite Telemetry Store (stonesense.db)\nfederated_rounds, hospital_training_runs, model_versions",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0f766e")

    gate_box = patches.FancyBboxPatch((6.2, 0.2), 5.2, 0.7, boxstyle="round,pad=0.03,rounding_size=0.05",
                                      facecolor="#fff1f2", edgecolor="#f43f5e", linewidth=1.2, zorder=2)
    ax.add_patch(gate_box)
    ax.text(8.8, 0.55, "Deployment Safety Gate (deployment_gate.py)\nAutomated Safety Verification: Eligible / Rejected / Pending",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#be123c")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig_8_1_federated_system_topology.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_8_2():
    """Figure 8.2: Sequence of one federated round using the 13 events in federated_coordinator.py."""
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 15)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Title
    ax.text(6.0, 14.5, "Figure 8.2: Execution Sequence & 13 Lifecycle Events in a Live Federated Round",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")
    ax.text(6.0, 14.15, "Synchronous Telemetry Flow Defined in backend/app/services/federated_coordinator.py",
            ha="center", va="center", fontsize=9.5, style="italic", color="#475569")

    # Timeline columns
    # Col 1: Developer Console / Admin (x=1.8)
    # Col 2: Central Coordinator (x=5.2)
    # Col 3: Hospital Clients 1-3 (x=8.8)
    # Col 4: Telemetry Store / Gate (x=11.0)
    actors = [
        ("Admin / Console", 1.8, "#6366f1"),
        ("FL Coordinator", 5.0, "#0284c7"),
        ("Hospital Nodes (1-3)", 8.5, "#16a34a"),
        ("Database & Gate", 11.0, "#9333ea")
    ]

    for title, x_pos, col in actors:
        # Box header
        box = patches.FancyBboxPatch((x_pos - 0.9, 13.2), 1.8, 0.6, boxstyle="round,pad=0.02,rounding_size=0.05",
                                     facecolor="#f8fafc", edgecolor=col, linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x_pos, 13.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col, zorder=4)
        # Lifeline
        ax.plot([x_pos, x_pos], [0.6, 13.2], color="#cbd5e1", linestyle="--", lw=1.2, zorder=1)

    events_seq = [
        (1, "ROUND_STARTED", 5.0, 1.8, "Coordinator initializes round across selected hospitals", 12.6, "<-"),
        (2, "GLOBAL_MODEL_DISTRIBUTING", 5.0, 8.5, "Coordinator distributes previous global checkpoint weights", 11.7, "->"),
        (3, "MODEL_RECEIVED", 8.5, 5.0, "Hospital nodes acknowledge receipt of global weights", 10.8, "<-"),
        (4, "HOSPITAL_TRAINING_STARTED", 8.5, 8.5, "Hospital nodes set state to TRAINING", 9.9, "self"),
        (5, "HOSPITAL_TRAINING", 8.5, 8.5, "Local optimizers train on isolated patient CT slices", 9.0, "self"),
        (6, "HOSPITAL_TRAINING_COMPLETED", 8.5, 8.5, "Local metrics computed (samples, acc, F1, loss)", 8.1, "self"),
        (7, "CLIENT_UPDATE_RECEIVED", 8.5, 5.0, "Hospital nodes transmit model parameters (NumPy arrays)", 7.2, "<-"),
        (8, "HOSPITAL_WAITING_FOR_AGGREGATION", 8.5, 8.5, "Nodes enter wait state awaiting peer client uploads", 6.3, "self"),
        (9, "FEDAVG_STARTED", 5.0, 5.0, "Coordinator initiates sample-weighted FedAvg aggregation", 5.4, "self"),
        (10, "FEDAVG_COMPLETED", 5.0, 1.8, "Aggregation finished; global validation metrics computed", 4.5, "<-"),
        (11, "MODEL_GATE_EVALUATED", 5.0, 11.0, "Candidate evaluated against clinical thresholds in safety gate", 3.6, "->"),
        (12, "GLOBAL_MODEL_SAVED", 5.0, 11.0, "Checkpoint persisted to disk; federated_rounds DB row saved", 2.7, "->"),
        (13, "ROUND_COMPLETED", 5.0, 1.8, "Round success synchronized to WebSocket console & hospitals", 1.8, "<-")
    ]

    for ev_num, ev_name, x_src, x_dst, ev_desc, y_val, arrow_dir in events_seq:
        # Event badge
        badge = patches.Circle((0.5, y_val), 0.22, facecolor="#0284c7", edgecolor="white", zorder=3)
        ax.add_patch(badge)
        ax.text(0.5, y_val, str(ev_num), ha="center", va="center", fontsize=8.5, fontweight="bold", color="white", zorder=4)

        # Event label & description
        ax.text(0.9, y_val + 0.1, ev_name, ha="left", va="center", fontsize=9, fontweight="bold", color="#0f172a")
        ax.text(0.9, y_val - 0.15, ev_desc, ha="left", va="center", fontsize=7.5, style="italic", color="#475569")

        # Message arrow on timeline
        if arrow_dir == "self":
            # Self loop arrow
            arc = patches.Arc((x_src + 0.25, y_val), 0.5, 0.4, angle=0, theta1=270, theta2=90, color="#64748b", lw=1.2)
            ax.add_patch(arc)
            ax.annotate("", xy=(x_src, y_val - 0.2), xytext=(x_src + 0.2, y_val - 0.2),
                        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
        elif arrow_dir == "->":
            ax.annotate("", xy=(x_dst, y_val), xytext=(x_src, y_val),
                        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5))
        elif arrow_dir == "<-":
            ax.annotate("", xy=(x_dst, y_val), xytext=(x_src, y_val),
                        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5))

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig_8_2_federated_round_event_sequence.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_8_3():
    """Figure 8.3: Global train loss and validation accuracy/F1 per round from stonesense.db."""
    conn = sqlite3.connect("stonesense.db")
    df = pd.read_sql_query(
        "SELECT round_number, global_train_loss, global_val_acc, global_val_f1 FROM federated_rounds ORDER BY round_number",
        conn
    )
    conn.close()

    # Filter out rows with None or incomplete metrics
    df = df.dropna(subset=['global_val_acc', 'global_train_loss'])
    rounds = df['round_number'].values
    train_loss = df['global_train_loss'].values
    val_acc = df['global_val_acc'].values * 100.0
    val_f1 = df['global_val_f1'].values * 100.0

    fig, ax1 = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    fig.patch.set_facecolor('white')

    color_loss = '#dc2626'
    color_acc = '#2563eb'
    color_f1 = '#059669'

    # Left Y-Axis: Global Train Loss
    ax1.set_xlabel('Federated Round Number', fontsize=10.5, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Global Train Loss (Cross-Entropy)', color=color_loss, fontsize=10.5, fontweight='bold')
    l1, = ax1.plot(rounds, train_loss, color=color_loss, marker='o', markersize=4.5, lw=1.8, label='Global Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_ylim(-0.02, 0.52)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    # Right Y-Axis: Global Validation Metrics
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy & Macro F1 (%)', color='#0f172a', fontsize=10.5, fontweight='bold')
    l2, = ax2.plot(rounds, val_acc, color=color_acc, marker='s', markersize=4.5, lw=1.8, label='Validation Accuracy (VALIDATION)')
    l3, = ax2.plot(rounds, val_f1, color=color_f1, marker='^', markersize=4.5, lw=1.8, linestyle='--', label='Validation Macro F1 (VALIDATION)')
    ax2.tick_params(axis='y', labelcolor='#0f172a')
    ax2.set_ylim(95.0, 100.5)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    # Annotation box for Rounds 1-3
    r1_acc = val_acc[0]
    r2_acc = val_acc[1]
    r3_acc = val_acc[2]
    annot_text = (
        f"Early Convergence (Validation Partition):\n"
        f"• Round 1: Val Acc = {r1_acc:.2f}%\n"
        f"• Round 2: Val Acc = {r2_acc:.2f}%\n"
        f"• Round 3: Val Acc = {r3_acc:.2f}%"
    )
    ax2.text(3.5, 96.2, annot_text, fontsize=8.5, fontweight='bold', color='#1e293b',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.0))

    # Combined legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9)

    fig.suptitle("Figure 8.3: Federated Convergence Telemetry per Round (Telemetry Store: stonesense.db)",
                 fontsize=12, fontweight='bold', y=0.98, color='#0f172a')

    fig.text(0.5, 0.01,
             "*Crucial Verification: All reported accuracy and F1 metrics are evaluated strictly on the combined multi-hospital VALIDATION partitions,\n"
             "NOT the held-out test split. Source: federated_rounds table in stonesense.db across 29 live rounds.",
             ha='center', fontsize=8, style='italic', color='#475569')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_file = OUTPUT_DIR / "fig_8_3_federated_training_convergence.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


def generate_fig_8_4():
    """Figure 8.4: Deployment gate flowchart showing only checks in backend/app/services/deployment_gate.py."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Header
    ax.text(6.0, 7.6, "Automated Deployment Gate Evaluation Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(6.0, 7.25, "Clinical Safety Verification Pipeline (backend/app/services/deployment_gate.py)",
            ha="center", va="center", fontsize=10, style="italic", color="#475569")

    # Input Node
    in_box = patches.FancyBboxPatch((4.2, 6.2), 3.6, 0.75, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor="#f8fafc", edgecolor="#475569", linewidth=1.5, zorder=2)
    ax.add_patch(in_box)
    ax.text(6.0, 6.57, "Candidate Model Evaluation Input", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax.text(6.0, 6.35, "candidate_metrics & validation_dataset_hash", ha="center", va="center", fontsize=8, color="#475569")

    ax.annotate("", xy=(6.0, 5.5), xytext=(6.0, 6.2), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))

    # Sequential Gate Checks Container
    gate_container = patches.Rectangle((0.6, 2.2), 10.8, 3.3, facecolor="#f8fafc", edgecolor="#cbd5e1", linestyle="--", lw=1.2, zorder=1)
    ax.add_patch(gate_container)
    ax.text(0.8, 5.3, "DEPLOYMENT GATE SAFETY RULES ENGINE", fontsize=9, fontweight="bold", color="#64748b")

    checks = [
        ("Check 1: Global Accuracy", "cand_acc >= 0.95", "min_accuracy threshold", 0.9, 3.7),
        ("Check 2: Macro F1-Score", "cand_f1 >= 0.94", "min_f1 threshold", 4.5, 3.7),
        ("Check 3: Stone Sensitivity", "recall_stone >= 0.96", "min_recall_stone threshold", 8.1, 3.7),
        ("Check 4: Tumor Sensitivity", "recall_tumor >= 0.98", "min_recall_tumor threshold", 0.9, 2.4),
        ("Check 5: Precision Monitor", "Informational FP Tracker", "Cyst/Normal -> Stone FP", 4.5, 2.4),
        ("Check 6: Regression Delta", "perf_drop <= 0.02", "Against active deployed model", 8.1, 2.4)
    ]

    for title, rule, desc, x_c, y_c in checks:
        cb = patches.FancyBboxPatch((x_c, y_c), 3.0, 1.1, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor="#ffffff", edgecolor="#3b82f6", linewidth=1.2, zorder=2)
        ax.add_patch(cb)
        ax.text(x_c + 1.5, y_c + 0.85, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")
        ax.text(x_c + 1.5, y_c + 0.55, rule, ha="center", va="center", fontsize=9, fontweight="bold", color="#dc2626")
        ax.text(x_c + 1.5, y_c + 0.25, desc, ha="center", va="center", fontsize=7.5, color="#64748b")

    # Branching arrows down from container
    ax.annotate("", xy=(2.4, 1.5), xytext=(2.4, 2.2), arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.0))
    ax.annotate("", xy=(6.0, 1.5), xytext=(6.0, 2.2), arrowprops=dict(arrowstyle="->", color="#eab308", lw=2.0))
    ax.annotate("", xy=(9.6, 1.5), xytext=(9.6, 2.2), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.0))

    # Outcomes
    # Outcome 1: Eligible
    o1 = patches.FancyBboxPatch((0.8, 0.4), 3.2, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=1.8, zorder=2)
    ax.add_patch(o1)
    ax.text(2.4, 1.15, "Status: ELIGIBLE", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#15803d")
    ax.text(2.4, 0.8, "All 6 checks successfully pass\nEligible for clinical deployment", ha="center", va="center", fontsize=8, color="#166534")

    # Outcome 2: Pending Review
    o2 = patches.FancyBboxPatch((4.4, 0.4), 3.2, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#fefce8", edgecolor="#ca8a04", linewidth=1.8, zorder=2)
    ax.add_patch(o2)
    ax.text(6.0, 1.15, "Status: PENDING_REVIEW", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#a16207")
    ax.text(6.0, 0.8, "Missing metrics or non-central\nvalidation partition signature", ha="center", va="center", fontsize=8, color="#854d0e")

    # Outcome 3: Rejected
    o3 = patches.FancyBboxPatch((8.0, 0.4), 3.2, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#fff1f2", edgecolor="#dc2626", linewidth=1.8, zorder=2)
    ax.add_patch(o3)
    ax.text(9.6, 1.15, "Status: REJECTED", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#b91c1c")
    ax.text(9.6, 0.8, "Any threshold failure or\nregression delta > 0.02 baseline", ha="center", va="center", fontsize=8, color="#991b1b")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig_8_4_deployment_gate_flowchart.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


if __name__ == "__main__":
    generate_fig_8_1()
    generate_fig_8_2()
    generate_fig_8_3()
    generate_fig_8_4()
