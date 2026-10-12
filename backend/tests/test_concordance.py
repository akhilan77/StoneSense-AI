"""Unit and API integration tests for the evidence concordance engine."""

import math
import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
from app.db.models import Hospital, Patient, Prediction
from app.schemas.concordance import (
    ConcordanceOutcomeCode,
    ConcordanceRequest,
    ConcordanceResult,
    ImagingOutputInput,
    TabularRiskInput,
)
from app.services.concordance_service import evaluate_concordance


# =============================================================================
# UNIT TESTS: Concordance Outcomes & Rule Precedence
# =============================================================================

def test_concordant_stone_evidence():
    """Elevated tabular risk AND CT predicts Stone -> CONCORDANT_STONE_EVIDENCE."""
    risk = TabularRiskInput(probability=0.82, risk_level="High", confidence=0.82)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.94)
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code == ConcordanceOutcomeCode.CONCORDANT_STONE_EVIDENCE
    assert res.rule_fired == "RULE_4_CONCORDANT_STONE_EVIDENCE"
    assert res.clinician_review_recommended is True
    assert "concordant" in res.outcome_label.lower()
    assert res.disclaimer


def test_concordant_no_stone_evidence():
    """Low tabular risk AND CT predicts Normal -> CONCORDANT_NO_STONE_EVIDENCE with required disclosures."""
    risk = TabularRiskInput(probability=0.18, risk_level="Low", confidence=0.82)
    imaging = ImagingOutputInput(class_name="Normal", confidence=0.88)
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code == ConcordanceOutcomeCode.CONCORDANT_NO_STONE_EVIDENCE
    assert res.outcome_label == "No stone evidence in either stream"
    assert res.rule_fired == "RULE_5_CONCORDANT_NO_STONE_EVIDENCE"
    assert res.clinician_review_recommended is False

    # Must explicitly state NOT a clearance, cite verified Stone recall and small tabular dataset size
    explanation_lower = res.explanation.lower()
    assert "not a clearance" in explanation_lower
    assert "65.70%" in res.explanation or "80.95%" in res.explanation
    assert "79" in res.explanation


def test_legacy_outcome_code_deserialization():
    """Safe backward compatibility: CONCORDANT_LOW_CONCERN deserializes to CONCORDANT_NO_STONE_EVIDENCE."""
    result = ConcordanceResult(
        outcome_code="CONCORDANT_LOW_CONCERN",
        outcome_label="No stone evidence in either stream",
        explanation="Legacy test explanation",
        rule_fired="RULE_5_CONCORDANT_NO_STONE_EVIDENCE",
        clinician_review_recommended=False,
        disclaimer="Disclaimer",
    )
    assert result.outcome_code == ConcordanceOutcomeCode.CONCORDANT_NO_STONE_EVIDENCE


def test_discordant_elevated_risk_normal_ct():
    """Elevated tabular risk AND CT predicts Normal -> DISCORDANT."""
    risk = TabularRiskInput(probability=0.75, risk_level="High", confidence=0.75)
    imaging = ImagingOutputInput(class_name="Normal", confidence=0.85)
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code == ConcordanceOutcomeCode.DISCORDANT
    assert res.rule_fired == "RULE_6_DISCORDANT_EVIDENCE"
    assert res.clinician_review_recommended is True


def test_discordant_low_risk_stone_ct():
    """Low tabular risk AND CT predicts Stone -> DISCORDANT."""
    risk = TabularRiskInput(probability=0.20, risk_level="Low", confidence=0.80)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.91)
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code == ConcordanceOutcomeCode.DISCORDANT
    assert res.rule_fired == "RULE_6_DISCORDANT_EVIDENCE"
    assert res.clinician_review_recommended is True


def test_non_stone_finding_tumor():
    """CT predicts Tumor -> NON_STONE_FINDING escalating to clinician review."""
    risk = TabularRiskInput(probability=0.10, risk_level="Low", confidence=0.90)
    imaging = ImagingOutputInput(class_name="Tumor", confidence=0.96)
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code == ConcordanceOutcomeCode.NON_STONE_FINDING
    assert res.rule_fired == "RULE_2_NON_STONE_FINDING"
    assert res.clinician_review_recommended is True
    assert "Tumor" in res.outcome_label


def test_non_stone_finding_cyst():
    """CT predicts Cyst -> NON_STONE_FINDING escalating to clinician review."""
    risk = TabularRiskInput(probability=0.80, risk_level="High", confidence=0.80)
    imaging = ImagingOutputInput(class_name="Cyst", confidence=0.89)
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code == ConcordanceOutcomeCode.NON_STONE_FINDING
    assert res.rule_fired == "RULE_2_NON_STONE_FINDING"
    assert res.clinician_review_recommended is True
    assert "Cyst" in res.outcome_label


def test_low_confidence_tabular():
    """Tabular confidence below threshold (0.60) triggers LOW_CONFIDENCE."""
    risk = TabularRiskInput(probability=0.52, risk_level="High", confidence=0.52)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.90)
    res = evaluate_concordance(risk=risk, imaging=imaging, tabular_min_confidence=0.60)

    assert res.outcome_code == ConcordanceOutcomeCode.LOW_CONFIDENCE
    assert res.rule_fired == "RULE_3_LOW_CONFIDENCE"
    assert res.clinician_review_recommended is True


def test_low_confidence_imaging():
    """Imaging confidence below threshold (0.60) triggers LOW_CONFIDENCE."""
    risk = TabularRiskInput(probability=0.80, risk_level="High", confidence=0.80)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.45)
    res = evaluate_concordance(risk=risk, imaging=imaging, imaging_min_confidence=0.60)

    assert res.outcome_code == ConcordanceOutcomeCode.LOW_CONFIDENCE
    assert res.rule_fired == "RULE_3_LOW_CONFIDENCE"
    assert res.clinician_review_recommended is True


def test_low_confidence_ood_gate_status():
    """Imaging flagged with status 'needs_review' triggers LOW_CONFIDENCE."""
    risk = TabularRiskInput(probability=0.75, risk_level="High", confidence=0.75)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.85, status="needs_review")
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code == ConcordanceOutcomeCode.LOW_CONFIDENCE
    assert res.rule_fired == "RULE_3_LOW_CONFIDENCE"


def test_insufficient_data_missing_inputs():
    """Missing risk or imaging input triggers INSUFFICIENT_DATA."""
    risk = TabularRiskInput(probability=0.70, risk_level="High", confidence=0.70)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.85)

    res_no_imaging = evaluate_concordance(risk=risk, imaging=None)
    assert res_no_imaging.outcome_code == ConcordanceOutcomeCode.INSUFFICIENT_DATA
    assert res_no_imaging.rule_fired == "RULE_1_INSUFFICIENT_DATA"

    res_no_risk = evaluate_concordance(risk=None, imaging=imaging)
    assert res_no_risk.outcome_code == ConcordanceOutcomeCode.INSUFFICIENT_DATA

    res_neither = evaluate_concordance(risk=None, imaging=None)
    assert res_neither.outcome_code == ConcordanceOutcomeCode.INSUFFICIENT_DATA


def test_insufficient_data_invalid_fields():
    """Invalid probability or unknown class triggers INSUFFICIENT_DATA."""
    valid_risk = TabularRiskInput(probability=0.60, risk_level="High", confidence=0.60)
    valid_imaging = ImagingOutputInput(class_name="Stone", confidence=0.80)

    res_bad_class = evaluate_concordance(risk=valid_risk, imaging={"class_name": "Fracture", "confidence": 0.90})
    assert res_bad_class.outcome_code == ConcordanceOutcomeCode.INSUFFICIENT_DATA

    res_nan_prob = evaluate_concordance(risk={"probability": float("nan")}, imaging=valid_imaging)
    assert res_nan_prob.outcome_code == ConcordanceOutcomeCode.INSUFFICIENT_DATA

    res_oob_prob = evaluate_concordance(risk={"probability": 1.5}, imaging=valid_imaging)
    assert res_oob_prob.outcome_code == ConcordanceOutcomeCode.INSUFFICIENT_DATA


# =============================================================================
# PRECEDENCE RULES TESTS
# =============================================================================

def test_precedence_insufficient_over_non_stone():
    """INSUFFICIENT_DATA takes precedence over NON_STONE_FINDING."""
    res = evaluate_concordance(risk=None, imaging={"class_name": "Tumor", "confidence": 0.95})
    assert res.outcome_code == ConcordanceOutcomeCode.INSUFFICIENT_DATA
    assert res.rule_fired == "RULE_1_INSUFFICIENT_DATA"


def test_precedence_non_stone_over_low_confidence():
    """NON_STONE_FINDING takes precedence over LOW_CONFIDENCE.

    Even if model confidence is low, a Tumor finding must ALWAYS escalate to clinician review.
    """
    risk = TabularRiskInput(probability=0.20, risk_level="Low", confidence=0.30)
    imaging = ImagingOutputInput(class_name="Tumor", confidence=0.35)
    res = evaluate_concordance(risk=risk, imaging=imaging, tabular_min_confidence=0.60, imaging_min_confidence=0.60)

    assert res.outcome_code == ConcordanceOutcomeCode.NON_STONE_FINDING
    assert res.rule_fired == "RULE_2_NON_STONE_FINDING"
    assert res.clinician_review_recommended is True


def test_precedence_low_confidence_over_concordant():
    """LOW_CONFIDENCE takes precedence over CONCORDANT_STONE_EVIDENCE."""
    risk = TabularRiskInput(probability=0.85, risk_level="High", confidence=0.40)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.90)
    res = evaluate_concordance(risk=risk, imaging=imaging, tabular_min_confidence=0.60)

    assert res.outcome_code == ConcordanceOutcomeCode.LOW_CONFIDENCE
    assert res.rule_fired == "RULE_3_LOW_CONFIDENCE"


# =============================================================================
# BOUNDARY VALUE & OVERRIDE TESTS
# =============================================================================

def test_boundary_risk_threshold_exact():
    """Exact risk threshold (0.50): boundary is inclusive of elevated risk."""
    risk_at = TabularRiskInput(probability=0.50, risk_level="Moderate", confidence=0.70)
    risk_below = TabularRiskInput(probability=0.4999, risk_level="Low", confidence=0.70)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.85)

    res_at = evaluate_concordance(risk=risk_at, imaging=imaging, risk_threshold=0.50)
    assert res_at.outcome_code == ConcordanceOutcomeCode.CONCORDANT_STONE_EVIDENCE

    res_below = evaluate_concordance(risk=risk_below, imaging=imaging, risk_threshold=0.50)
    assert res_below.outcome_code == ConcordanceOutcomeCode.DISCORDANT


def test_boundary_confidence_threshold_exact():
    """Confidence exactly at threshold (0.60) is sufficient and does not trigger LOW_CONFIDENCE."""
    risk = TabularRiskInput(probability=0.80, risk_level="High", confidence=0.60)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.60)

    res_exact = evaluate_concordance(
        risk=risk, imaging=imaging, tabular_min_confidence=0.60, imaging_min_confidence=0.60
    )
    assert res_exact.outcome_code == ConcordanceOutcomeCode.CONCORDANT_STONE_EVIDENCE

    res_below = evaluate_concordance(
        risk=TabularRiskInput(probability=0.80, risk_level="High", confidence=0.599),
        imaging=imaging,
        tabular_min_confidence=0.60,
    )
    assert res_below.outcome_code == ConcordanceOutcomeCode.LOW_CONFIDENCE


def test_custom_disclaimer_and_threshold_overrides():
    """Covers custom disclaimer string and explicit threshold arguments."""
    risk = TabularRiskInput(probability=0.70, risk_level="High", confidence=0.80)
    imaging = ImagingOutputInput(class_name="Stone", confidence=0.85)
    custom_disc = "Custom clinical research disclaimer text"

    res = evaluate_concordance(
        risk=risk,
        imaging=imaging,
        risk_threshold=0.65,
        tabular_min_confidence=0.75,
        imaging_min_confidence=0.75,
        disclaimer_text=custom_disc,
    )
    assert res.disclaimer == custom_disc
    assert res.thresholds_applied["risk_threshold"] == 0.65
    assert res.thresholds_applied["tabular_min_confidence"] == 0.75
    assert res.thresholds_applied["imaging_min_confidence"] == 0.75


# =============================================================================
# REGRESSION TEST: Cyst/Tumor NEVER yields No Stone Evidence
# =============================================================================

@pytest.mark.parametrize("prob_step", [i * 0.05 for i in range(21)])
@pytest.mark.parametrize("non_stone_class", ["Cyst", "Tumor", "CYST", "TUMOR"])
def test_regression_non_stone_never_no_stone_evidence(prob_step, non_stone_class):
    """Asserts Cyst or Tumor never yields CONCORDANT_NO_STONE_EVIDENCE under any tabular probability."""
    risk = TabularRiskInput(probability=prob_step, risk_level="Low", confidence=0.95)
    imaging = ImagingOutputInput(class_name=non_stone_class, confidence=0.95)
    res = evaluate_concordance(risk=risk, imaging=imaging)

    assert res.outcome_code != ConcordanceOutcomeCode.CONCORDANT_NO_STONE_EVIDENCE
    assert res.outcome_code != ConcordanceOutcomeCode.CONCORDANT_LOW_CONCERN
    assert res.outcome_code == ConcordanceOutcomeCode.NON_STONE_FINDING
    assert res.clinician_review_recommended is True


# =============================================================================
# API INTEGRATION TESTS: /api/v1/predict/concordance
# =============================================================================

def test_api_concordance_with_stored_ids(client: TestClient, db, auth_headers_hospital_1):
    """POST /api/v1/predict/concordance accepts server-stored prediction references."""
    # Create server-stored predictions for Hospital 1
    risk_pred = Prediction(
        hospital_id=1,
        prediction_type="risk",
        model_name="logistic_regression_v001",
        result_label="High",
        confidence=0.88,
        latency_ms=12.5,
    )
    img_pred = Prediction(
        hospital_id=1,
        prediction_type="image",
        model_name="ResNet18",
        result_label="Stone",
        confidence=0.92,
        latency_ms=45.0,
    )
    db.add(risk_pred)
    db.add(img_pred)
    db.commit()
    db.refresh(risk_pred)
    db.refresh(img_pred)

    payload = {
        "risk_prediction_id": risk_pred.id,
        "image_prediction_id": img_pred.id,
    }
    res = client.post("/api/v1/predict/concordance", json=payload, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    data = res.json()
    assert data["outcome_code"] == "CONCORDANT_STONE_EVIDENCE"
    assert data["rule_fired"] == "RULE_4_CONCORDANT_STONE_EVIDENCE"


def test_api_concordance_cross_tenant_denied(client: TestClient, db, auth_headers_hospital_1):
    """Cross-tenant access to another hospital's prediction reference must be denied with 403."""
    foreign_pred = Prediction(
        hospital_id=2,  # Belongs to Hospital 2
        prediction_type="risk",
        model_name="logistic_regression_v001",
        result_label="High",
        confidence=0.88,
    )
    img_pred = Prediction(
        hospital_id=1,  # Belongs to Hospital 1
        prediction_type="image",
        model_name="ResNet18",
        result_label="Stone",
        confidence=0.92,
    )
    db.add(foreign_pred)
    db.add(img_pred)
    db.commit()
    db.refresh(foreign_pred)
    db.refresh(img_pred)

    payload = {
        "risk_prediction_id": foreign_pred.id,
        "image_prediction_id": img_pred.id,
    }
    # Hospital 1 user requests Hospital 2 prediction
    res = client.post("/api/v1/predict/concordance", json=payload, headers=auth_headers_hospital_1)
    assert res.status_code == 403
    assert "Cross-tenant access denied" in res.json()["detail"]


def test_api_concordance_patient_mismatch_denied(client: TestClient, db, auth_headers_hospital_1):
    """Prediction reference that belongs to a different patient must be denied with 403."""
    risk_pred = Prediction(
        hospital_id=1,
        patient_id=10,
        prediction_type="risk",
        model_name="logistic_regression_v001",
        result_label="Low",
        confidence=0.85,
    )
    db.add(risk_pred)
    db.commit()
    db.refresh(risk_pred)

    payload = {
        "risk_prediction_id": risk_pred.id,
        "patient_id": 99,  # Mismatched patient ID
    }
    res = client.post("/api/v1/predict/concordance", json=payload, headers=auth_headers_hospital_1)
    assert res.status_code == 403
    assert "Patient authorization mismatch" in res.json()["detail"]


def test_api_concordance_direct_payload_disabled_by_default(client: TestClient, auth_headers_hospital_1, monkeypatch):
    """Direct client-supplied probabilities are rejected by default with 400 for clinical safety."""
    monkeypatch.setattr(settings, "allow_direct_concordance_payload", False)

    payload = {
        "risk": {"probability": 0.84, "risk_level": "High", "confidence": 0.84},
        "imaging": {"class_name": "Stone", "confidence": 0.95},
    }
    res = client.post("/api/v1/predict/concordance", json=payload, headers=auth_headers_hospital_1)
    assert res.status_code == 400
    assert "deprecated and disabled" in res.json()["detail"]


def test_api_concordance_direct_payload_when_enabled(client: TestClient, auth_headers_hospital_1, monkeypatch):
    """Direct client-supplied probabilities succeed when allow_direct_concordance_payload is enabled."""
    monkeypatch.setattr(settings, "allow_direct_concordance_payload", True)

    payload = {
        "risk": {"probability": 0.84, "risk_level": "High", "confidence": 0.84},
        "imaging": {"class_name": "Stone", "confidence": 0.95},
    }
    res = client.post("/api/v1/predict/concordance", json=payload, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    assert res.json()["outcome_code"] == "CONCORDANT_STONE_EVIDENCE"


def test_api_concordance_unauthorized(client: TestClient):
    """POST /api/v1/predict/concordance returns 401 without JWT authentication."""
    payload = {"risk_prediction_id": 1, "image_prediction_id": 2}
    res = client.post("/api/v1/predict/concordance", json=payload)
    assert res.status_code == 401


def test_api_concordance_bad_payload(client: TestClient, auth_headers_hospital_1):
    """POST /api/v1/predict/concordance returns 422 on schema violation (e.g. probability > 1.0)."""
    payload = {
        "risk": {"probability": 99.5},
        "imaging": {"class_name": "Stone"},
    }
    res = client.post("/api/v1/predict/concordance", json=payload, headers=auth_headers_hospital_1)
    assert res.status_code == 422
