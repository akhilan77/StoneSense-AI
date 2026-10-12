"""Pure, deterministic evidence concordance engine.

Evaluates independent outputs from the tabular urine biochemistry risk model
and the deep learning CT scan slice classification model without synthetic data fusion.
"""

from typing import Any, Dict, Optional, Union
import math

from app.config.settings import settings
from app.schemas.concordance import (
    ConcordanceOutcomeCode,
    ConcordanceRequest,
    ConcordanceResult,
    ImagingOutputInput,
    TabularRiskInput,
)

KNOWN_CT_CLASSES = {"CYST", "NORMAL", "STONE", "TUMOR"}


def evaluate_concordance(
    risk: Optional[Union[TabularRiskInput, Dict[str, Any]]] = None,
    imaging: Optional[Union[ImagingOutputInput, Dict[str, Any]]] = None,
    risk_threshold: Optional[float] = None,
    tabular_min_confidence: Optional[float] = None,
    imaging_min_confidence: Optional[float] = None,
    disclaimer_text: Optional[str] = None,
) -> ConcordanceResult:
    """Pure, deterministic concordance evaluation between independent clinical models.

    Rule Precedence Order:
        1. RULE_1_INSUFFICIENT_DATA: Missing or invalid inputs in either stream.
        2. RULE_2_NON_STONE_FINDING: CT predicts Cyst or Tumor. ALWAYS escalates
           to clinician review regardless of tabular risk or confidence.
        3. RULE_3_LOW_CONFIDENCE: Either stream has confidence below its threshold,
           or imaging review status was rejected/needs_review.
        4. RULE_4_CONCORDANT_STONE_EVIDENCE: Elevated tabular risk AND CT predicts Stone.
        5. RULE_5_CONCORDANT_LOW_CONCERN: Low tabular risk AND CT predicts Normal.
        6. RULE_6_DISCORDANT_EVIDENCE: Streams point in opposite directions.

    Args:
        risk: Tabular risk prediction input (or dict).
        imaging: CT imaging prediction input (or dict).
        risk_threshold: Overridable risk probability cutoff (default from settings).
        tabular_min_confidence: Overridable min confidence for tabular stream.
        imaging_min_confidence: Overridable min confidence for imaging stream.
        disclaimer_text: Overridable regulatory disclaimer string.

    Returns:
        ConcordanceResult: Structured deterministic concordance outcome.
    """
    applied_risk_thresh = (
        float(risk_threshold)
        if risk_threshold is not None
        else float(settings.concordance_risk_threshold)
    )
    applied_tab_min_conf = (
        float(tabular_min_confidence)
        if tabular_min_confidence is not None
        else float(settings.concordance_tabular_min_confidence)
    )
    applied_img_min_conf = (
        float(imaging_min_confidence)
        if imaging_min_confidence is not None
        else float(settings.concordance_imaging_min_confidence)
    )
    applied_disclaimer = (
        disclaimer_text
        if disclaimer_text is not None
        else settings.concordance_disclaimer
    )

    thresholds_applied = {
        "risk_threshold": applied_risk_thresh,
        "tabular_min_confidence": applied_tab_min_conf,
        "imaging_min_confidence": applied_img_min_conf,
    }

    # Normalize tabular inputs
    norm_risk: Optional[Dict[str, Any]] = None
    if risk is not None:
        if isinstance(risk, TabularRiskInput):
            norm_risk = risk.model_dump()
        elif isinstance(risk, dict):
            norm_risk = dict(risk)

    # Normalize imaging inputs
    norm_imaging: Optional[Dict[str, Any]] = None
    if imaging is not None:
        if isinstance(imaging, ImagingOutputInput):
            norm_imaging = imaging.model_dump()
        elif isinstance(imaging, dict):
            norm_imaging = dict(imaging)

    inputs_used = {
        "risk": norm_risk,
        "imaging": norm_imaging,
    }

    # -------------------------------------------------------------------------
    # RULE 1: INSUFFICIENT DATA
    # -------------------------------------------------------------------------
    if norm_risk is None or norm_imaging is None:
        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.INSUFFICIENT_DATA,
            outcome_label="Insufficient Data",
            explanation="One or both assessment streams are missing. Both urine biochemistry and CT imaging analyses must be completed before evaluating multi-stream concordance.",
            rule_fired="RULE_1_INSUFFICIENT_DATA",
            clinician_review_recommended=False,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    prob = norm_risk.get("probability")
    risk_level = norm_risk.get("risk_level")
    tab_conf = norm_risk.get("confidence")

    class_name = norm_imaging.get("class_name")
    img_conf = norm_imaging.get("confidence")
    img_status = norm_imaging.get("status")

    # Validate numerical boundaries and validity
    def _is_invalid_num(val: Any) -> bool:
        if val is None:
            return True
        try:
            f = float(val)
            return math.isnan(f) or math.isinf(f)
        except (ValueError, TypeError):
            return True

    if (prob is None and risk_level is None) or class_name is None:
        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.INSUFFICIENT_DATA,
            outcome_label="Insufficient Data",
            explanation="Assessment stream values are incomplete. Missing tabular probability/risk or CT classification label.",
            rule_fired="RULE_1_INSUFFICIENT_DATA",
            clinician_review_recommended=False,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    if prob is not None and (_is_invalid_num(prob) or not (0.0 <= float(prob) <= 1.0)):
        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.INSUFFICIENT_DATA,
            outcome_label="Insufficient Data",
            explanation=f"Invalid tabular risk probability value: {prob}.",
            rule_fired="RULE_1_INSUFFICIENT_DATA",
            clinician_review_recommended=False,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    normalized_class = str(class_name).strip().upper()
    if normalized_class not in KNOWN_CT_CLASSES:
        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.INSUFFICIENT_DATA,
            outcome_label="Insufficient Data",
            explanation=f"Unknown CT classification class '{class_name}'. Expected one of {sorted(list(KNOWN_CT_CLASSES))}.",
            rule_fired="RULE_1_INSUFFICIENT_DATA",
            clinician_review_recommended=False,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    # -------------------------------------------------------------------------
    # RULE 2: NON-STONE FINDING (CYST / TUMOR)
    # Highest priority clinical override: must NEVER be softened by tabular low risk
    # -------------------------------------------------------------------------
    if normalized_class in ("CYST", "TUMOR"):
        finding = "Cyst" if normalized_class == "CYST" else "Tumor"
        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.NON_STONE_FINDING,
            outcome_label=f"Clinician Review: Non-Stone Finding ({finding})",
            explanation=(
                f"CT imaging identified an incidental or alternative renal finding ({finding}). "
                f"Regardless of urine biochemistry risk, non-stone findings require direct clinician "
                f"evaluation and definitive radiological follow-up."
            ),
            rule_fired="RULE_2_NON_STONE_FINDING",
            clinician_review_recommended=True,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    # -------------------------------------------------------------------------
    # RULE 3: LOW CONFIDENCE / OUT-OF-DISTRIBUTION WARNING
    # -------------------------------------------------------------------------
    # Derive effective confidence for tabular if not directly provided
    eff_tab_conf = (
        float(tab_conf)
        if tab_conf is not None and not _is_invalid_num(tab_conf)
        else (max(float(prob), 1.0 - float(prob)) if prob is not None else 1.0)
    )
    eff_img_conf = (
        float(img_conf)
        if img_conf is not None and not _is_invalid_num(img_conf)
        else 0.0
    )

    is_ood_flagged = img_status in ("needs_review", "rejected_ood")
    if (
        eff_tab_conf < applied_tab_min_conf
        or eff_img_conf < applied_img_min_conf
        or is_ood_flagged
    ):
        reasons = []
        if eff_tab_conf < applied_tab_min_conf:
            reasons.append(
                f"tabular model confidence ({eff_tab_conf:.2f} < {applied_tab_min_conf:.2f})"
            )
        if eff_img_conf < applied_img_min_conf:
            reasons.append(
                f"CT model confidence ({eff_img_conf:.2f} < {applied_img_min_conf:.2f})"
            )
        if is_ood_flagged:
            reasons.append(f"CT image flagged by review gate ({img_status})")

        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.LOW_CONFIDENCE,
            outcome_label="Low Confidence Assessment",
            explanation=(
                f"One or both evidence streams exhibit low certainty ({', '.join(reasons)}). "
                f"Automated concordance cannot be established reliably; manual clinical inspection is required."
            ),
            rule_fired="RULE_3_LOW_CONFIDENCE",
            clinician_review_recommended=True,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    # Determine tabular elevated vs low
    is_tabular_elevated = False
    if prob is not None:
        is_tabular_elevated = float(prob) >= applied_risk_thresh
    elif risk_level is not None:
        is_tabular_elevated = str(risk_level).strip().lower() in ("high", "moderate")

    # -------------------------------------------------------------------------
    # RULE 4: CONCORDANT STONE EVIDENCE
    # -------------------------------------------------------------------------
    if is_tabular_elevated and normalized_class == "STONE":
        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.CONCORDANT_STONE_EVIDENCE,
            outcome_label="Concordant Stone Evidence",
            explanation=(
                "Both urine biochemistry profile and axial CT imaging independently present evidence "
                "consistent with nephrolithiasis. Cross-validation across independent modalities increases "
                "diagnostic confidence, though decision support remains subject to clinician verification."
            ),
            rule_fired="RULE_4_CONCORDANT_STONE_EVIDENCE",
            clinician_review_recommended=True,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    # -------------------------------------------------------------------------
    # RULE 5: CONCORDANT LOW CONCERN
    # -------------------------------------------------------------------------
    if not is_tabular_elevated and normalized_class == "NORMAL":
        return ConcordanceResult(
            outcome_code=ConcordanceOutcomeCode.CONCORDANT_LOW_CONCERN,
            outcome_label="Concordant Low Concern",
            explanation=(
                "Both urine biochemistry profile (low risk score) and CT scan slice (normal renal parenchyma) "
                "independently demonstrate low concern for acute nephrolithiasis."
            ),
            rule_fired="RULE_5_CONCORDANT_LOW_CONCERN",
            clinician_review_recommended=False,
            inputs_used=inputs_used,
            thresholds_applied=thresholds_applied,
            disclaimer=applied_disclaimer,
        )

    # -------------------------------------------------------------------------
    # RULE 6: DISCORDANT EVIDENCE
    # -------------------------------------------------------------------------
    tabular_desc = (
        f"elevated risk (prob: {float(prob):.1%})"
        if is_tabular_elevated
        else f"low risk (prob: {float(prob):.1%})"
    ) if prob is not None else (f"{risk_level} risk")

    ct_desc = f"{normalized_class.capitalize()} ({eff_img_conf:.1%})"

    return ConcordanceResult(
        outcome_code=ConcordanceOutcomeCode.DISCORDANT,
        outcome_label="Discordant Multi-Stream Evidence",
        explanation=(
            f"The two diagnostic modalities point in differing directions: urine biochemistry indicates "
            f"{tabular_desc}, whereas CT imaging indicates {ct_desc}. "
            f"Such discrepancy can occur with non-obstructive radiopaque stones or metabolic abnormalities "
            f"without visible calculi, and warrants physician case review."
        ),
        rule_fired="RULE_6_DISCORDANT_EVIDENCE",
        clinician_review_recommended=True,
        inputs_used=inputs_used,
        thresholds_applied=thresholds_applied,
        disclaimer=applied_disclaimer,
    )
