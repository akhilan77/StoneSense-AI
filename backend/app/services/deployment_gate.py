"""Deployment Gate Evaluation Service for StoneSense-AI DL Models.

Evaluates candidate federated/centralized DL model checkpoints against
configured clinical safety criteria, per-class sensitivity thresholds,
and regression baselines prior to production deployment eligibility.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.db.models import ModelVersion

import hashlib
from pathlib import Path

logger = logging.getLogger("DeploymentGate")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def compute_validation_dataset_hash(val_paths: Optional[List[Any]] = None) -> str:
    """Computes a SHA256 signature hash of the sorted relative files in the validation set."""
    if not val_paths:
        val_paths = [PROJECT_ROOT / "dl" / "datasets" / "partitions" / f"hospital_{i}" / "validation" for i in [1, 2, 3]]

    hasher = hashlib.sha256()
    files = []
    for vp in val_paths:
        vp = Path(vp)
        if vp.exists():
            for f in vp.rglob("*.*"):
                if f.is_file():
                    files.append(f.relative_to(vp).as_posix() + f":{f.stat().st_size}")
    files.sort()
    for entry in files:
        hasher.update(entry.encode("utf-8"))
    return hasher.hexdigest()[:16] if files else "empty_dataset"


class DeploymentGate:
    """Automated quality and safety gate for ResNet18 federated models."""

    def __init__(self, config=None):
        self.config = config or settings

    def evaluate_candidate(
        self,
        candidate_metrics: Dict[str, Any],
        db: Session,
        model_family: str = "resnet18_ct",
        validation_data_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates candidate model metrics against safety thresholds and active deployed baseline.

        Rules:
        - Must satisfy min_accuracy, min_f1.
        - Must include measured recall_stone and recall_tumor satisfying min_recall thresholds.
          If per-class metrics are missing, check is marked 'not_evaluated' and status becomes 'pending_review'.
        - Compares against active deployed model on identical validation set.
          If data source differs, regression check is 'not_comparable' -> 'pending_review'.
          If regression exceeds max_regression_vs_current, check fails -> 'rejected'.
          If no deployed model exists, regression check is 'skipped'.
        - If any check fails threshold -> 'rejected'.
        - If all checks pass -> 'eligible'.
        """
        data_source = (
            validation_data_source
            or candidate_metrics.get("validation_data_source")
            or candidate_metrics.get("data_source")
            or compute_validation_dataset_hash()
        )

        checks: List[Dict[str, Any]] = []
        has_failure = False
        has_pending_review = False
        reasons: List[str] = []

        # 1. Global Accuracy Check
        cand_acc = candidate_metrics.get("accuracy")
        if cand_acc is not None:
            cand_acc = float(cand_acc)
            acc_passed = cand_acc >= self.config.gate_min_accuracy
            checks.append({
                "name": "min_accuracy",
                "label": "Global Accuracy",
                "value": round(cand_acc, 4),
                "threshold": self.config.gate_min_accuracy,
                "operator": ">=",
                "status": "passed" if acc_passed else "failed",
                "details": f"Accuracy {cand_acc:.4f} vs minimum threshold {self.config.gate_min_accuracy:.4f}",
            })
            if not acc_passed:
                has_failure = True
                reasons.append(
                    f"Accuracy ({cand_acc:.4f}) fell below minimum required threshold ({self.config.gate_min_accuracy:.4f})."
                )
        else:
            checks.append({
                "name": "min_accuracy",
                "label": "Global Accuracy",
                "value": None,
                "threshold": self.config.gate_min_accuracy,
                "operator": ">=",
                "status": "not_evaluated",
                "details": "Global accuracy metric is missing.",
            })
            has_pending_review = True
            reasons.append("Missing global accuracy metric prevents automatic eligibility.")

        # 2. Global Macro F1 Check
        cand_f1 = candidate_metrics.get("f1") if "f1" in candidate_metrics else candidate_metrics.get("f1_score") if "f1_score" in candidate_metrics else candidate_metrics.get("f1_macro")
        if cand_f1 is not None:
            cand_f1 = float(cand_f1)
            f1_passed = cand_f1 >= self.config.gate_min_f1
            checks.append({
                "name": "min_f1",
                "label": "Macro F1 Score",
                "value": round(cand_f1, 4),
                "threshold": self.config.gate_min_f1,
                "operator": ">=",
                "status": "passed" if f1_passed else "failed",
                "details": f"Macro F1 {cand_f1:.4f} vs minimum threshold {self.config.gate_min_f1:.4f}",
            })
            if not f1_passed:
                has_failure = True
                reasons.append(
                    f"Macro F1 ({cand_f1:.4f}) fell below minimum required threshold ({self.config.gate_min_f1:.4f})."
                )
        else:
            checks.append({
                "name": "min_f1",
                "label": "Macro F1 Score",
                "value": None,
                "threshold": self.config.gate_min_f1,
                "operator": ">=",
                "status": "not_evaluated",
                "details": "Macro F1 metric is missing.",
            })
            has_pending_review = True
            reasons.append("Missing macro F1 metric prevents automatic eligibility.")

        # 3. Per-Class Recall (Stone) Check
        cand_rec_stone = candidate_metrics.get("recall_stone")
        if cand_rec_stone is None and "per_class_metrics" in candidate_metrics:
            stone_meta = candidate_metrics["per_class_metrics"].get("Stone") or candidate_metrics["per_class_metrics"].get("stone")
            if isinstance(stone_meta, dict):
                cand_rec_stone = stone_meta.get("recall")

        if cand_rec_stone is not None:
            cand_rec_stone = float(cand_rec_stone)
            rec_stone_passed = cand_rec_stone >= self.config.gate_min_recall_stone
            checks.append({
                "name": "min_recall_stone",
                "label": "Stone Recall Sensitivity",
                "value": round(cand_rec_stone, 4),
                "threshold": self.config.gate_min_recall_stone,
                "operator": ">=",
                "status": "passed" if rec_stone_passed else "failed",
                "details": f"Stone recall {cand_rec_stone:.4f} vs threshold {self.config.gate_min_recall_stone:.4f}",
            })
            if not rec_stone_passed:
                has_failure = True
                reasons.append(
                    f"Stone recall sensitivity ({cand_rec_stone:.4f}) fell below required clinical threshold ({self.config.gate_min_recall_stone:.4f})."
                )
        else:
            checks.append({
                "name": "min_recall_stone",
                "label": "Stone Recall Sensitivity",
                "value": None,
                "threshold": self.config.gate_min_recall_stone,
                "operator": ">=",
                "status": "not_evaluated",
                "details": "Per-class Stone recall not measured.",
            })
            has_pending_review = True
            reasons.append("Per-class Stone recall was not evaluated; manual review required before eligibility.")

        # 4. Per-Class Recall (Tumor) Check
        cand_rec_tumor = candidate_metrics.get("recall_tumor")
        if cand_rec_tumor is None and "per_class_metrics" in candidate_metrics:
            tumor_meta = candidate_metrics["per_class_metrics"].get("Tumor") or candidate_metrics["per_class_metrics"].get("tumor")
            if isinstance(tumor_meta, dict):
                cand_rec_tumor = tumor_meta.get("recall")

        if cand_rec_tumor is not None:
            cand_rec_tumor = float(cand_rec_tumor)
            rec_tumor_passed = cand_rec_tumor >= self.config.gate_min_recall_tumor
            checks.append({
                "name": "min_recall_tumor",
                "label": "Tumor Recall Sensitivity",
                "value": round(cand_rec_tumor, 4),
                "threshold": self.config.gate_min_recall_tumor,
                "operator": ">=",
                "status": "passed" if rec_tumor_passed else "failed",
                "details": f"Tumor recall {cand_rec_tumor:.4f} vs threshold {self.config.gate_min_recall_tumor:.4f}",
            })
            if not rec_tumor_passed:
                has_failure = True
                reasons.append(
                    f"Tumor recall sensitivity ({cand_rec_tumor:.4f}) fell below required clinical threshold ({self.config.gate_min_recall_tumor:.4f})."
                )
        else:
            checks.append({
                "name": "min_recall_tumor",
                "label": "Tumor Recall Sensitivity",
                "value": None,
                "threshold": self.config.gate_min_recall_tumor,
                "operator": ">=",
                "status": "not_evaluated",
                "details": "Per-class Tumor recall not measured.",
            })
            has_pending_review = True
            reasons.append("Per-class Tumor recall was not evaluated; manual review required before eligibility.")

        # 5. Informational: Stone Precision & False Positive Confusion (Cyst/Normal -> Stone)
        cand_prec_stone = candidate_metrics.get("precision_stone")
        if cand_prec_stone is None and "per_class_metrics" in candidate_metrics:
            stone_meta = candidate_metrics["per_class_metrics"].get("Stone") or candidate_metrics["per_class_metrics"].get("stone")
            if isinstance(stone_meta, dict):
                cand_prec_stone = stone_meta.get("precision")

        cm = candidate_metrics.get("confusion_matrix")
        cyst_to_stone = None
        normal_to_stone = None
        if isinstance(cm, list) and len(cm) >= 3 and len(cm[0]) >= 3:
            cyst_to_stone = int(cm[0][2])
            normal_to_stone = int(cm[1][2])

        checks.append({
            "name": "stone_precision_info",
            "label": "Stone Precision & Confusion (Informational)",
            "value": {
                "precision_stone": round(float(cand_prec_stone), 4) if cand_prec_stone is not None else None,
                "cyst_confused_as_stone": cyst_to_stone,
                "normal_confused_as_stone": normal_to_stone,
            },
            "threshold": None,
            "operator": "informational",
            "status": "info",
            "details": (
                f"Informational monitor: Stone precision = "
                f"{round(float(cand_prec_stone), 4) if cand_prec_stone is not None else 'N/A'}, "
                f"Cyst->Stone FP = {cyst_to_stone if cyst_to_stone is not None else 'N/A'}, "
                f"Normal->Stone FP = {normal_to_stone if normal_to_stone is not None else 'N/A'}."
            ),
        })

        # 6. Regression Check against Currently Deployed Model
        deployed_model = (
            db.query(ModelVersion)
            .filter(ModelVersion.is_deployed.is_(True), ModelVersion.model_family == model_family)
            .first()
        )

        if not deployed_model:
            checks.append({
                "name": "regression_check",
                "label": "Regression vs Active Deployed",
                "value": None,
                "threshold": self.config.gate_max_regression_vs_current,
                "operator": "<=",
                "status": "skipped",
                "details": "No existing active deployed model in registry to compare against.",
            })
        else:
            # Check validation set identity and central evaluation basis
            deployed_gate = deployed_model.gate_report or {}
            deployed_ds = deployed_gate.get("data_source") or deployed_gate.get("validation_set_hash") or self.config.gate_validation_data_source
            is_central = deployed_gate.get("is_central_eval", True)

            if not is_central or deployed_ds != data_source:
                checks.append({
                    "name": "regression_check",
                    "label": "Regression vs Active Deployed",
                    "value": None,
                    "threshold": self.config.gate_max_regression_vs_current,
                    "operator": "<=",
                    "status": "not_comparable",
                    "details": (
                        f"Validation dataset mismatch or non-central baseline: candidate signature '{data_source}' "
                        f"vs deployed signature '{deployed_ds}' (central_eval={is_central})."
                    ),
                })
                has_pending_review = True
                reasons.append(
                    f"Candidate evaluated on '{data_source}' while deployed model used '{deployed_ds}' (central={is_central}). Manual review required."
                )
            else:
                # Compare accuracy and F1 regression
                dep_acc = deployed_model.accuracy or 0.0
                dep_f1 = deployed_model.f1_score or 0.0
                max_reg = self.config.gate_max_regression_vs_current

                acc_drop = max(0.0, dep_acc - (cand_acc if cand_acc is not None else 0.0))
                f1_drop = max(0.0, dep_f1 - (cand_f1 if cand_f1 is not None else 0.0))

                reg_passed = (acc_drop <= max_reg) and (f1_drop <= max_reg)
                checks.append({
                    "name": "regression_check",
                    "label": "Regression vs Active Deployed",
                    "value": {
                        "accuracy_drop": round(acc_drop, 4),
                        "f1_drop": round(f1_drop, 4),
                        "deployed_version": deployed_model.version_tag,
                    },
                    "threshold": max_reg,
                    "operator": "<=",
                    "status": "passed" if reg_passed else "failed",
                    "details": (
                        f"Active {deployed_model.version_tag} (Acc: {dep_acc:.4f}, F1: {dep_f1:.4f}). "
                        f"Observed drops: Acc -{acc_drop:.4f}, F1 -{f1_drop:.4f} (Max allowed: {max_reg:.4f})"
                    ),
                })
                if not reg_passed:
                    has_failure = True
                    reasons.append(
                        f"Candidate model regressed against active deployed model {deployed_model.version_tag} "
                        f"(Acc drop: {acc_drop:.4f}, F1 drop: {f1_drop:.4f}, limit: {max_reg:.4f})."
                    )

        # 7. Verification & Data Leakage Audit Check
        is_leaky = candidate_metrics.get("trained_on_leaky_partitions", False)
        metrics_void = candidate_metrics.get("metrics_void", False)
        patient_verified = candidate_metrics.get("patient_level_separation_verified", True)
        is_unverified = candidate_metrics.get("is_unverified", False)

        if is_leaky or metrics_void:
            checks.append({
                "name": "data_leakage_audit",
                "label": "Data Leakage & Integrity Audit",
                "value": "leaky_partitions",
                "threshold": "zero_leakage",
                "operator": "==",
                "status": "failed",
                "details": "Candidate trained on unverified or leaky partitions; promotion rejected.",
            })
            has_failure = True
            reasons.append("Model trained on unverified or leaky partitions cannot be promoted as validated.")
        elif not patient_verified or is_unverified:
            checks.append({
                "name": "patient_level_verification",
                "label": "Patient-Level Verification",
                "value": "unverified",
                "threshold": "verified",
                "operator": "==",
                "status": "pending_review",
                "details": "Patient-level separation remains unverified (no patient IDs); requires manual clinical review.",
            })
            has_pending_review = True
            reasons.append("Patient-level separation remains unverified; manual review required before eligibility.")
        else:
            checks.append({
                "name": "audit_verification",
                "label": "Integrity & Patient Separation Audit",
                "value": "verified",
                "threshold": "verified",
                "operator": "==",
                "status": "passed",
                "details": "Audit integrity and split isolation verified.",
            })

        # Determine overall gate status
        if has_failure:
            final_status = "rejected"
            passed = False
        elif has_pending_review:
            final_status = "pending_review"
            passed = False
        else:
            final_status = "eligible"
            passed = True

        gate_report = {
            "passed": passed,
            "status": final_status,
            "model_family": model_family,
            "data_source": data_source,
            "reasons": reasons,
            "checks": checks,
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Deployment Gate Evaluated: Status={final_status} | "
            f"Passed={passed} | Checks={len(checks)} | Reasons={len(reasons)}"
        )
        return gate_report


deployment_gate = DeploymentGate()
