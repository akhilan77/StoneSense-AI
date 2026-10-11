"""Script to generate Figure 9.1: Entity-Relationship (ER) Diagram from backend/app/db/models.py (12 tables).
Outputs docs/report_figures/fig_9_1_er_diagram.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Georgia', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = Path("docs/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_table_card(ax, x, y, w, h, table_name, pk_list, fk_list, attr_list, header_color="#1e3a8a", bg_color="#f8fafc"):
    """Helper to draw an academic ER table entity card with PK, FK, and Attributes."""
    # Outer box
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=bg_color,
        edgecolor=header_color,
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(box)

    # Header banner
    hdr_h = 0.35
    hdr = patches.FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=header_color,
        edgecolor=header_color,
        linewidth=1.2,
        zorder=3
    )
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - hdr_h / 2, table_name, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=4)

    # Content
    y_cur = y + h - hdr_h - 0.16
    line_h = 0.15

    for pk in pk_list:
        ax.text(x + 0.1, y_cur, f"[PK] {pk}", fontsize=7.5, fontweight="bold", color="#b91c1c", zorder=4)
        y_cur -= line_h

    for fk in fk_list:
        ax.text(x + 0.1, y_cur, f"[FK] {fk}", fontsize=7.2, fontweight="bold", color="#0369a1", zorder=4)
        y_cur -= line_h

    for attr in attr_list:
        ax.text(x + 0.1, y_cur, f"• {attr}", fontsize=7.0, color="#334155", zorder=4)
        y_cur -= line_h


def generate_fig_9_1():
    """Generates Figure 9.1: Comprehensive Relational ER Diagram for 12 Database Tables."""
    fig, ax = plt.subplots(figsize=(15, 10.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Overall Title
    ax.text(7.5, 10.15, "StoneSense-AI Relational Telemetry & Operational Database Schema (stonesense.db)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(7.5, 9.85, "12 SQLAlchemy Relational Entities (Defined in backend/app/db/models.py) with Cardinality Relations",
            ha="center", va="center", fontsize=9.5, style="italic", color="#475569")

    # Row 1 (Top): Identity, Tenancy & Security
    # 1. hospitals
    draw_table_card(ax, 0.6, 7.3, 3.2, 2.3, "hospitals (Tenancy / Clients)",
                    ["id: Integer (PK)"],
                    [],
                    ["hospital_code: String(50) [UQ]", "name: String(255)", "region: String(100)",
                     "is_active: Boolean", "created_at: DateTime", "updated_at: DateTime"],
                    "#1e3a8a", "#eff6ff")

    # 2. users
    draw_table_card(ax, 4.3, 7.3, 3.2, 2.3, "users (RBAC Identity)",
                    ["id: Integer (PK)"],
                    ["hospital_id: FK -> hospitals.id"],
                    ["email: String(255) [UQ]", "hashed_password: String", "role: Enum(admin, dev, hosp)",
                     "full_name: String", "is_active: Boolean", "created_at: DateTime"],
                    "#1e3a8a", "#eff6ff")

    # 3. audit_logs
    draw_table_card(ax, 8.0, 7.3, 3.2, 2.3, "audit_logs (Security Trace)",
                    ["id: Integer (PK)"],
                    ["user_id: FK -> users.id"],
                    ["action: String(100)", "resource: String(100)", "details: Text / JSON",
                     "ip_address: String(45)", "created_at: DateTime"],
                    "#475569", "#f8fafc")

    # 4. system_logs
    draw_table_card(ax, 11.6, 7.3, 2.8, 2.3, "system_logs (Operations)",
                    ["id: Integer (PK)"],
                    [],
                    ["component: String(100)", "level: String(20)", "message: Text",
                     "metadata_json: Text", "created_at: DateTime"],
                    "#475569", "#f8fafc")

    # Row 2 (Middle): Federated Learning & Model Governance
    # 5. federated_rounds
    draw_table_card(ax, 0.6, 4.1, 3.3, 2.7, "federated_rounds (FL Telemetry)",
                    ["id: Integer (PK)"],
                    [],
                    ["round_number: Integer [UQ]", "status: String(50)", "participants_count: Int",
                     "global_train_loss: Float", "global_train_acc: Float", "global_val_loss: Float",
                     "global_val_acc: Float", "global_val_f1: Float", "selected_hospital_ids: Text",
                     "duration_sec: Float", "completed_at: DateTime"],
                    "#047857", "#f0fdf4")

    # 6. hospital_training_runs
    draw_table_card(ax, 4.4, 4.1, 3.2, 2.7, "hospital_training_runs (Local Runs)",
                    ["id: Integer (PK)"],
                    ["round_id: FK -> federated_rounds.id", "hospital_id: FK -> hospitals.id"],
                    ["local_train_loss: Float", "local_train_acc: Float", "local_val_acc: Float",
                     "local_val_f1: Float", "sample_count: Int", "status: String(50)",
                     "completed_at: DateTime"],
                    "#047857", "#f0fdf4")

    # 7. model_versions
    draw_table_card(ax, 8.1, 4.1, 3.2, 2.7, "model_versions (Model Registry)",
                    ["id: Integer (PK)"],
                    ["round_id: FK -> federated_rounds.id"],
                    ["version_tag: String(100) [UQ]", "model_family: String(50)", "checkpoint_path: String",
                     "accuracy: Float", "f1_score: Float", "stone_recall: Float",
                     "tumor_recall: Float", "is_deployed: Boolean", "status: String(50)",
                     "gate_report: JSON"],
                    "#047857", "#f0fdf4")

    # 8. hospital_update_logs
    draw_table_card(ax, 11.6, 4.1, 2.8, 2.7, "hospital_update_logs (Uploads)",
                    ["id: Integer (PK)"],
                    ["hospital_id: FK -> hospitals.id", "round_id: FK -> federated_rounds.id"],
                    ["update_type: String(50)", "payload_hash: String(64)", "duration_sec: Float",
                     "timestamp: DateTime"],
                    "#047857", "#f0fdf4")

    # Row 3 (Bottom): Clinical Patient Care, Inferences & Drift
    # 9. patients
    draw_table_card(ax, 0.6, 0.8, 3.2, 2.7, "patients (Clinical Cohort)",
                    ["id: Integer (PK)"],
                    ["hospital_id: FK -> hospitals.id"],
                    ["patient_code: String(50) [UQ]", "encrypted_name: String", "age: Integer",
                     "gender: String(20)", "clinical_notes: Text", "created_at: DateTime"],
                    "#7c2d12", "#fef2f2")

    # 10. predictions
    draw_table_card(ax, 4.3, 0.8, 3.3, 2.7, "predictions (Diagnostic Logs)",
                    ["id: Integer (PK)"],
                    ["patient_id: FK -> patients.id", "hospital_id: FK -> hospitals.id", "model_version_id: FK -> model_versions.id"],
                    ["prediction_type: String(50)", "predicted_class: String(50)", "confidence: Float",
                     "probabilities: JSON", "shap_values: JSON", "gradcam_path: String",
                     "created_at: DateTime"],
                    "#7c2d12", "#fef2f2")

    # 11. inference_logs
    draw_table_card(ax, 8.1, 0.8, 3.1, 2.7, "inference_logs (Latency & SLO)",
                    ["id: Integer (PK)"],
                    ["model_version_id: FK -> model_versions.id", "hospital_id: FK -> hospitals.id"],
                    ["latency_ms: Float", "status_code: Integer", "route: String(100)",
                     "timestamp: DateTime"],
                    "#7c2d12", "#fef2f2")

    # 12. drift_records
    draw_table_card(ax, 11.6, 0.8, 2.8, 2.7, "drift_records (Data Quality)",
                    ["id: Integer (PK)"],
                    ["hospital_id: FK -> hospitals.id"],
                    ["metric_name: String(50)", "drift_score: Float", "p_value: Float",
                     "is_drift_detected: Boolean", "detected_at: DateTime"],
                    "#7c2d12", "#fef2f2")

    # Relationship connector lines & arrows
    connectors = [
        # hospitals -> users
        ((3.8, 8.5), (4.3, 8.5)),
        # users -> audit_logs
        ((7.5, 8.5), (8.0, 8.5)),
        # hospitals -> hospital_training_runs
        ((2.2, 7.3), (5.0, 6.8)),
        # federated_rounds -> hospital_training_runs
        ((3.9, 5.5), (4.4, 5.5)),
        # federated_rounds -> model_versions
        ((3.9, 5.0), (8.1, 5.0)),
        # hospitals -> patients
        ((1.5, 7.3), (1.5, 3.5)),
        # patients -> predictions
        ((3.8, 2.2), (4.3, 2.2)),
        # model_versions -> predictions
        ((8.5, 4.1), (6.5, 3.5)),
        # model_versions -> inference_logs
        ((9.6, 4.1), (9.6, 3.5)),
        # hospitals -> drift_records
        ((3.0, 7.3), (11.6, 2.2))
    ]

    for (x1, y1), (x2, y2) in connectors:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2, linestyle=":"))

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig_9_1_er_diagram.png"
    plt.savefig(out_file, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Generated {out_file}")


if __name__ == "__main__":
    generate_fig_9_1()
