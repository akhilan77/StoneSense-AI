"""Unit tests for AssessmentService explainability aggregation routines."""

import pytest
from app.services.assessment_service import AssessmentService


def test_assessment_service_build():
    """Verifies that build_assessment orchestrates tabular predictions and SHAP feature analysis."""
    dummy_data = {
        "gravity": 1.015,
        "ph": 6.2,
        "osmo": 550,
        "cond": 22.0,
        "urea": 250,
        "calc": 4.5
    }
    # Test without image uploads first
    res = AssessmentService.build_assessment(patient_data=dummy_data)
    
    assert "risk_prediction" in res
    assert "recommendation" in res
    assert "shap" in res
    assert "top_features" in res["shap"]
    assert "calc" in res["shap"]["top_features"]
    assert "ct_prediction" in res
    assert res["ct_prediction"] == {}
