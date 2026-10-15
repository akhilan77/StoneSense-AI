"""FastAPI API routes implementing real model predictions, complete assessments, and explainability endpoints."""

import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.core.auth import get_current_user, get_scoped_hospital_id, log_audit
from app.db.database import get_db
from app.db.models import InferenceLog, Patient, Prediction, User
from app.schemas.concordance import (
    ConcordanceRequest,
    ConcordanceResult,
    ImagingOutputInput,
    TabularRiskInput,
)
from app.schemas.nhanes_risk import NhanesCompareRequest, NhanesCompareResponse
from app.schemas.patient import PatientInformation
from app.schemas.responses import (
    RiskPredictionResponse,
    StoneDetectionResponse,
)
from app.services.concordance_service import evaluate_concordance
from app.services.explainability_service import (
    generate_gradcam_for_bytes,
    generate_shap_for_patient,
)
from app.services.model_loader import model_loader
from app.services.nhanes_compare_service import (
    FeatureValidationError,
    ModelsUnavailableError,
    nhanes_compare_service,
)
from app.services.prediction_service import PredictionService
from app.utils.image_utils import (
    evaluate_ct_ood_gate,
    read_bounded_image_bytes,
    validate_ct_image_content,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/concordance", response_model=ConcordanceResult)
async def predict_concordance(
    payload: ConcordanceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConcordanceResult:
    """Evaluates transparent, rule-based evidence concordance between tabular and imaging streams.
    
    Accepts server-stored prediction references (risk_prediction_id, image_prediction_id)
    with strict tenant isolation and patient scoping. Direct client-supplied probabilities
    are deprecated and rejected unless allow_direct_concordance_payload is explicitly enabled.
    """
    effective_hospital_id = get_scoped_hospital_id(current_user, current_user.hospital_id)

    risk_input: Optional[TabularRiskInput] = None
    img_input: Optional[ImagingOutputInput] = None

    has_ref_ids = payload.risk_prediction_id is not None or payload.image_prediction_id is not None
    has_direct_payload = payload.risk is not None or payload.imaging is not None

    if has_ref_ids:
        # 1. Lookup server-stored risk prediction
        if payload.risk_prediction_id is not None:
            risk_pred = db.query(Prediction).filter(
                Prediction.id == payload.risk_prediction_id,
            ).first()
            if not risk_pred:
                raise HTTPException(status_code=404, detail=f"Risk prediction #{payload.risk_prediction_id} not found.")
            if risk_pred.prediction_type != "risk":
                raise HTTPException(status_code=400, detail=f"Prediction #{payload.risk_prediction_id} is not a risk prediction.")

            # Tenant isolation
            if current_user.role != "admin" and risk_pred.hospital_id != effective_hospital_id:
                raise HTTPException(status_code=403, detail="Cross-tenant access denied: risk prediction belongs to a different hospital organization.")

            # Patient authorization
            if payload.patient_id is not None and risk_pred.patient_id is not None and risk_pred.patient_id != payload.patient_id:
                raise HTTPException(status_code=403, detail="Patient authorization mismatch: risk prediction does not belong to specified patient.")

            r_level = "High" if str(risk_pred.result_label).lower() == "high" else "Low"
            r_conf = float(risk_pred.confidence) if risk_pred.confidence is not None else 0.5
            r_prob = r_conf if r_level == "High" else (1.0 - r_conf)
            risk_input = TabularRiskInput(
                probability=r_prob,
                risk_level=r_level,
                confidence=r_conf,
            )

        # 2. Lookup server-stored imaging prediction
        if payload.image_prediction_id is not None:
            img_pred = db.query(Prediction).filter(
                Prediction.id == payload.image_prediction_id,
            ).first()
            if not img_pred:
                raise HTTPException(status_code=404, detail=f"Imaging prediction #{payload.image_prediction_id} not found.")
            if img_pred.prediction_type != "image":
                raise HTTPException(status_code=400, detail=f"Prediction #{payload.image_prediction_id} is not an imaging prediction.")

            # Tenant isolation
            if current_user.role != "admin" and img_pred.hospital_id != effective_hospital_id:
                raise HTTPException(status_code=403, detail="Cross-tenant access denied: imaging prediction belongs to a different hospital organization.")

            # Patient authorization
            if payload.patient_id is not None and img_pred.patient_id is not None and img_pred.patient_id != payload.patient_id:
                raise HTTPException(status_code=403, detail="Patient authorization mismatch: imaging prediction does not belong to specified patient.")

            c_label = img_pred.result_label or "Normal"
            c_conf = float(img_pred.confidence) if img_pred.confidence is not None else 0.5
            c_status = "needs_review" if c_label == "needs_review" else "accepted"
            img_input = ImagingOutputInput(
                class_name=c_label,
                confidence=c_conf,
                probabilities={c_label: c_conf},
                status=c_status,
            )
    elif has_direct_payload:
        if not settings.allow_direct_concordance_payload:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Direct client-supplied probabilities in /predict/concordance are deprecated and disabled "
                    "for clinical safety. Please supply server-stored prediction references (risk_prediction_id "
                    "and image_prediction_id). To enable direct payloads in testing environments, set "
                    "STONESENSE_ALLOW_DIRECT_CONCORDANCE_PAYLOAD=true."
                ),
            )
        risk_input = payload.risk
        img_input = payload.imaging

    result = evaluate_concordance(risk=risk_input, imaging=img_input)

    # Standard database audit log entry (non-cryptographic relational record)
    log_audit(
        db=db,
        action="PREDICTION_CONCORDANCE",
        resource_type="concordance",
        hospital_id=current_user.hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={
            "outcome_code": result.outcome_code.value,
            "rule_fired": result.rule_fired,
            "status": "evaluated",
        },
    )

    return result


@router.post("/risk", response_model=RiskPredictionResponse)
async def predict_risk(
    payload: PatientInformation,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RiskPredictionResponse:
    """Classifies patient kidney stone risk from urine biochemistry parameters."""
    if model_loader.ml_model is None:
        raise HTTPException(status_code=503, detail="Risk prediction model not loaded.")

    effective_hospital_id = get_scoped_hospital_id(current_user, payload.hospital_id)

    patient = None
    if payload.patient_id is not None:
        patient = db.query(Patient).filter(
            Patient.id == payload.patient_id,
            Patient.hospital_id == effective_hospital_id,
        ).first()
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found for this hospital.")

    start_time = time.time()
    res = PredictionService.predict_risk(payload.model_dump())
    elapsed = time.time() - start_time
    confidence = max(res["probability"], 1 - res["probability"])

    active_ml_ver = getattr(model_loader, "active_ml_version_tag", "logistic_regression_v001")
    active_ml_fam = getattr(model_loader, "active_ml_family", "logistic_regression")

    prediction_id = None
    try:
        prediction = Prediction(
            hospital_id=effective_hospital_id,
            patient_id=patient.id if patient else None,
            prediction_type="risk",
            model_name=active_ml_ver,
            result_label=str(res["risk"]),
            confidence=confidence,
            latency_ms=elapsed * 1000,
        )
        inf_log = InferenceLog(
            hospital_id=effective_hospital_id,
            prediction_type="risk",
            model_name=active_ml_fam,
            version_tag=active_ml_ver,
            result_label=str(res["risk"]),
            confidence=confidence,
            latency_ms=elapsed * 1000,
        )
        db.add(prediction)
        db.add(inf_log)
        if patient:
            profile = dict(patient.urine_features or {})
            profile["clinical"] = payload.model_dump(exclude={"hospital_id", "patient_id"})
            patient.urine_features = profile
        db.commit()
        db.refresh(prediction)
        prediction_id = prediction.id
    except Exception:
        db.rollback()

    log_audit(
        db=db,
        action="PREDICTION_RISK",
        resource_type="prediction",
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={
            "patient_id": patient.id if patient else None,
            "result_label": str(res["risk"]),
            "confidence": round(float(confidence), 4),
            "model_version": active_ml_ver,
        },
    )

    shap = generate_shap_for_patient(payload.model_dump())
    return RiskPredictionResponse(
        probability=res["probability"],
        risk_level=res["risk"],
        confidence=confidence,
        inference_time_sec=round(elapsed, 4),
        shap=shap,
        prediction_id=prediction_id,
    )


@router.get("/risk/compare/spec")
def nhanes_compare_spec(current_user: User = Depends(get_current_user)) -> dict:
    """Returns ml/artifacts/nhanes_v003/feature_spec.json: the definition the clinical-profile form is built from."""
    try:
        return nhanes_compare_service.get_spec()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/risk/compare", response_model=NhanesCompareResponse)
def predict_risk_compare(
    payload: NhanesCompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> NhanesCompareResponse:
    """Scores a clinical profile with the four NHANES v003b models (EBM, CatBoost, XGBoost, LightGBM).

    Returns each model's calibrated probability, risk level at the selected operating point ('accuracy' or 'balanced';
    both are returned per model with their thresholds and stored held-out TEST metrics), top contributing factors, plus the average probability and whether the models agree.
    Auth, RBAC and tenant scoping match /predict/risk. The submitted values are used for scoring only and are
    not stored (no schema change); only an audit-log entry without feature values is written.
    """
    effective_hospital_id = get_scoped_hospital_id(current_user, payload.hospital_id)

    patient = None
    if payload.patient_id is not None:
        patient = db.query(Patient).filter(
            Patient.id == payload.patient_id,
            Patient.hospital_id == effective_hospital_id,
        ).first()
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found for this hospital.")

    try:
        result = nhanes_compare_service.compare(payload.features, operating_point=payload.operating_point)
    except FeatureValidationError as exc:
        raise HTTPException(status_code=422, detail={"message": "Invalid clinical profile.", "problems": exc.problems})
    except (ModelsUnavailableError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    log_audit(
        db=db,
        action="PREDICTION_RISK_COMPARE",
        resource_type="prediction",
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={
            "patient_id": patient.id if patient else None,
            "prediction_type": "risk_compare",
            "model_version": "nhanes_v003b",
            "result_label": f"{result['n_high']}/4 models High; agree={result['models_agree']}",
            "confidence": result["average_probability"],
        },
    )
    return NhanesCompareResponse(**result)


@router.post("/image", response_model=StoneDetectionResponse)
async def predict_image(
    image: UploadFile = File(...),
    model_family: Optional[str] = Form("resnet18"),
    hospital_id: Optional[int] = Form(None),
    patient_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StoneDetectionResponse:
    """Classifies CT scan slice using selected DL model architecture (ResNet18, YOLO26, DINOv3, QKNN)."""
    from dl.models.registry import dl_registry

    selected_model_id = dl_registry.resolve_model_id(model_family or "resnet18")
    try:
        wrapper = dl_registry.get_model(selected_model_id)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_family '{model_family}'. Allowed: ['resnet18', 'yolo26', 'dinov3', 'qknn']"
        )

    if not wrapper.is_ready:
        if wrapper.status.value == "PENDING_WEIGHTS":
            raise HTTPException(
                status_code=503,
                detail=f"Model '{wrapper.model_name}' is currently unavailable (status: PENDING_WEIGHTS). "
                       f"{wrapper._error_message or 'Trained artifacts have not yet been deployed.'}"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{wrapper.model_name}' is not ready ({wrapper.status.value}): {wrapper._error_message}"
            )

    effective_hospital_id = get_scoped_hospital_id(current_user, hospital_id)

    patient = None
    if patient_id is not None:
        patient = db.query(Patient).filter(
            Patient.id == patient_id,
            Patient.hospital_id == effective_hospital_id,
        ).first()
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found for this hospital.")

    # 1. Bounded upload read to prevent unbounded memory consumption
    content = await read_bounded_image_bytes(image)

    # 2. Strict content verification (magic bytes, decompression limit, dimensions, sanity)
    _, image_warnings, sanity_metrics = validate_ct_image_content(content)

    start_time = time.time()
    try:
        res = PredictionService.predict_ct_image(content, model_id=selected_model_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image with model '{wrapper.model_name}': {str(e)}")

    elapsed = time.time() - start_time

    # 3. Out-of-distribution & low-confidence safety gate
    raw_probabilities = res.get("probabilities", {})
    gate_status, review_reason, ood_metrics = evaluate_ct_ood_gate(
        probabilities=raw_probabilities,
        confidence=res["confidence"],
        warnings=image_warnings,
    )

    output_dir = Path(__file__).resolve().parents[3] / "static" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = None
    overlay_url = None
    gradcam_error = None
    try:
        overlay = generate_gradcam_for_bytes(
            content,
            str(output_dir / f"{uuid4().hex}_overlay.png"),
            target_class=res["class"],
            model_id=selected_model_id
        )
        if overlay.get("overlay_path") and Path(overlay["overlay_path"]).exists():
            overlay_url = f"/static/gradcam/{Path(overlay['overlay_path']).name}"
    except Exception as exc:
        gradcam_error = str(exc)
        import logging
        logging.getLogger("predict_router").exception(f"Explainability generation failed for model '{wrapper.model_name}'")

    model_display_tag = f"{res['model_id']}_{res['class']}"
    recorded_family = res.get("model_family", f"{res['model_id']}_ct")

    # If gate fires, return 'needs_review' rather than confident diagnosis label
    effective_class_label = "needs_review" if gate_status == "needs_review" else res["class"]

    prediction_id = None
    try:
        prediction = Prediction(
            hospital_id=effective_hospital_id,
            patient_id=patient.id if patient else None,
            prediction_type="image",
            model_name=res["model_name"],
            result_label=str(effective_class_label),
            confidence=res["confidence"],
            latency_ms=elapsed * 1000,
            explainability_ref=overlay_url,
        )
        inf_log = InferenceLog(
            hospital_id=effective_hospital_id,
            prediction_type="image",
            model_name=recorded_family,
            version_tag=res["model_id"],
            result_label=str(effective_class_label),
            confidence=res["confidence"],
            latency_ms=elapsed * 1000,
        )
        db.add(prediction)
        db.add(inf_log)
        db.commit()
        db.refresh(prediction)
        prediction_id = prediction.id
    except Exception:
        db.rollback()

    log_audit(
        db=db,
        action="PREDICTION_CT",
        resource_type="prediction",
        hospital_id=effective_hospital_id,
        user_id=current_user.id,
        user_email=current_user.email,
        details={
            "patient_id": patient.id if patient else None,
            "result_label": str(effective_class_label),
            "confidence": round(float(res["confidence"]), 4),
            "model_id": res["model_id"],
            "model_name": res["model_name"],
            "status": gate_status,
            "ood_metrics": ood_metrics,
        },
    )

    payload = {
        "class_name": effective_class_label,
        "confidence": res["confidence"],
        "inference_time_sec": round(elapsed, 4),
        "model_id": res["model_id"],
        "model_name": res["model_name"],
        "status": gate_status,
        "review_reason": review_reason,
        "probabilities": raw_probabilities,
        "ood_metrics": ood_metrics,
        "warnings": image_warnings,
        "prediction_id": prediction_id,
    }
    if overlay:
        payload["gradcam"] = {
            "overlay_url": overlay_url if (overlay.get("available") and overlay_url) else "",
            "target_class": overlay.get("target_class", res["class"]),
            "available": bool(overlay.get("available", False)),
            "message": overlay.get("message", "")
        }
    elif gradcam_error:
        payload["gradcam"] = {
            "overlay_url": "",
            "target_class": res["class"],
            "available": False,
            "message": f"Explanation unavailable for {res['model_name']}."
        }

    return StoneDetectionResponse(**payload)

