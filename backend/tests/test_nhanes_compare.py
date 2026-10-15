"""Tests for the NHANES v003b four-model comparison: service validation, wrappers, registry and the HTTP endpoint.

Most tests use tiny models trained on synthetic data and registered in a temporary registry, so they do not depend
on the real notebook outputs. The last test class checks the real registered models when they exist.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.risk_models.nhanes_bundle import BANNED_COLUMNS, ProbCalibrator  # noqa: E402
from ml.risk_models.registry import ModelRegistry  # noqa: E402
from ml.risk_models.wrappers import (  # noqa: E402
    CatBoostRiskModel, EBMRiskModel, FAMILY_MODEL_MAP, LightGBMRiskModel, XGBoostNhanesRiskModel, XGBoostRiskModel,
)
from ml.training import nhanes_common as nc  # noqa: E402

from app.services.nhanes_compare_service import (  # noqa: E402
    FeatureValidationError, ModelsUnavailableError, NhanesCompareService, MODEL_TAGS,
)

FEATS = ["RIDAGEYR", "BMXBMI", "LBXSUA", "URXUMA", "URXUCR", "URDACT", "BPQ020", "RIDRETH1"]
VALID = {"RIDAGEYR": 50, "BMXBMI": 27.5, "LBXSUA": 5.5, "URXUMA": 8.0, "URXUCR": 110.0, "BPQ020": 0}


def _synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    age = rng.uniform(20, 80, n)
    bmi = rng.uniform(18, 45, n)
    uma = np.exp(rng.normal(2, 1, n))
    ucr = rng.uniform(20, 300, n)
    htn = (rng.rand(n) < 0.3).astype(float)
    logit = -3.2 + 0.02 * (age - 50) + 0.03 * (bmi - 28) + 0.4 * htn
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "RIDAGEYR": age, "BMXBMI": bmi, "LBXSUA": rng.uniform(3, 9, n), "URXUMA": uma, "URXUCR": ucr,
        "URDACT": uma / ucr * 100, "BPQ020": htn, "RIDRETH1": rng.randint(1, 6, n).astype(float), "target_stone": y,
    })


def _op(thr: float, acc: float, sens: float, spec: float) -> dict:
    return {"threshold": thr, "accuracy": acc, "accuracy_ci95": [acc - 0.02, acc + 0.02], "balanced_accuracy": (sens + spec) / 2,
            "balanced_accuracy_ci95": [0.55, 0.65], "sensitivity": sens, "sensitivity_ci95": [sens - 0.04, sens + 0.04],
            "specificity": spec, "specificity_ci95": [spec - 0.02, spec + 0.02], "positive_rate": 1 - spec, "tp": 1, "fp": 1, "tn": 1, "fn": 1}


OP_ACC = _op(0.20, 0.72, 0.46, 0.75)
OP_BAL = _op(0.12, 0.64, 0.60, 0.65)


def _test_metrics() -> dict:
    return {"evaluated_on": "test (2017-2018)", "test_cycle": "2017-2018", "n": 5558, "prevalence": 0.0997,
            "majority_baseline_accuracy": 0.9003, "roc_auc": 0.65, "roc_auc_ci95": [0.63, 0.67], "pr_auc": 0.17, "brier": 0.09,
            "at_accuracy_threshold": OP_ACC, "at_balanced_threshold": OP_BAL}


def _entry_fields() -> dict:
    return {"status": "pending_review", "variant": "base_history_sensitive", "test_metrics": _test_metrics(),
            "default_operating_point": "accuracy", "thresholds": {"accuracy": OP_ACC["threshold"], "balanced": OP_BAL["threshold"]},
            "imbalance_strategy": "none",
            "operating_points": {"accuracy": {"label": "Accuracy-focused (>70% accuracy)", "rule": "r", "threshold": OP_ACC["threshold"],
                                              "validation": {**OP_ACC, "accuracy": 0.7203}, "test": OP_ACC},
                                 "balanced": {"label": "Balanced (max balanced accuracy)", "rule": "r", "threshold": OP_BAL["threshold"],
                                              "validation": OP_BAL, "test": OP_BAL}}}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier

    tmp = tmp_path_factory.mktemp("nhanes_tiny")
    df = _synthetic()
    train, val = df.iloc[:1100], df.iloc[1100:]
    vd = nc.prepare_variant({"train": train, "validation": val}, "base_history_sensitive", FEATS)
    estimators = {
        "xgboost": xgb.XGBClassifier(n_estimators=30, max_depth=3, device="cpu", verbosity=0, random_state=0).fit(vd.X_train, vd.y_train),
        "lightgbm": lgb.LGBMClassifier(n_estimators=30, num_leaves=7, verbose=-1, random_state=0).fit(vd.X_train, vd.y_train),
        "catboost": CatBoostClassifier(iterations=30, depth=3, verbose=0, allow_writing_files=False, thread_count=1, random_seed=0).fit(vd.X_train, vd.y_train),
        "ebm": ExplainableBoostingClassifier(interactions=2, outer_bags=2, max_rounds=40, n_jobs=1, random_state=0).fit(vd.X_train, vd.y_train),
    }
    registry = ModelRegistry(versions_dir=tmp / "versions", manifest_path=tmp / "manifest.json")
    wrappers = {"xgboost": XGBoostNhanesRiskModel, "lightgbm": LightGBMRiskModel, "catboost": CatBoostRiskModel, "ebm": EBMRiskModel}
    for family, est in estimators.items():
        raw = est.predict_proba(vd.X_val)[:, 1]
        cal = ProbCalibrator.fit(raw, vd.y_val, "sigmoid")
        bundle = nc.make_bundle(f"{family}_nhanes_v003b", family, "base_history_sensitive", vd, est, cal,
                                {"balanced": OP_BAL["threshold"], "accuracy": OP_ACC["threshold"]})
        model = wrappers[family](bundle=bundle, version_tag=f"{family}_nhanes_v003b", threshold=bundle.threshold)
        registry.register_model(model, metadata={"metrics": {}}, extra_entry_fields=_entry_fields())
    cfg = {"primary_variant": "base_history_sensitive", "variants": {"base_history_sensitive": vd.features},
           "history_questions": {"BPQ020": "Has a doctor ever told you that you have high blood pressure?"}}
    spec = nc.build_feature_spec(cfg, train)
    spec_path = nc.save_json(spec, tmp / "feature_spec.json")
    service = NhanesCompareService(registry=registry, spec_path=spec_path)
    return {"service": service, "registry": registry, "vd": vd, "val": val}


# --------------------------------------------------------------------------- wrappers / registry
def test_family_map_has_new_wrappers():
    assert {"catboost", "lightgbm", "ebm", "xgboost"} <= set(FAMILY_MODEL_MAP)
    assert FAMILY_MODEL_MAP["catboost"] is CatBoostRiskModel and FAMILY_MODEL_MAP["ebm"] is EBMRiskModel
    assert FAMILY_MODEL_MAP["xgboost"] is XGBoostRiskModel   # v001/v002 wrapper unchanged


def test_registry_round_trip_and_standard_explanation_schema(tiny):
    reg, val = tiny["registry"], tiny["val"].head(3)
    for tag in MODEL_TAGS:
        reg2 = ModelRegistry(versions_dir=reg.versions_dir, manifest_path=reg.manifest_path)
        model = reg2.get_model(tag)
        probs = model.predict_proba(val)
        assert probs.shape == (3, 2) and np.allclose(probs.sum(axis=1), 1.0)
        assert ((probs >= 0) & (probs <= 1)).all()
        ex = model.explain(val.iloc[[0]])
        assert set(ex) >= {"top_features", "feature_contributions", "feature_directions", "summary"}
        assert set(ex["feature_contributions"]) == set(model.bundle.input_features)
        assert set(ex["feature_directions"].values()) <= {"increases", "decreases", "neutral"}


def test_banned_columns_absent_from_model_inputs(tiny):
    for tag in MODEL_TAGS:
        inputs = set(tiny["registry"].get_model(tag).bundle.input_features)
        assert not inputs & set(BANNED_COLUMNS)
        assert "KIQ029" not in inputs


def test_tree_shap_contributions_sum_to_margin(tiny):
    """Additivity check: SHAP contributions + base value reproduce the raw log-odds for a tree model."""
    model = tiny["registry"].get_model("xgboost_nhanes_v003b")
    b, rows = model.bundle, tiny["val"].head(5)
    contrib = b.contributions(rows).sum(axis=1)
    margin = b.estimator.predict(b.transform(rows), output_margin=True)
    base = float(np.mean(margin - contrib))
    assert np.allclose(margin - contrib, base, atol=1e-3)


def test_family_lookup_falls_back_to_bundles_only_when_no_legacy_model_exists(tiny):
    # the temporary registry holds only v003b bundles, so the family lookup may return one of them
    assert tiny["registry"].get_model_by_family("catboost").version_tag == "catboost_nhanes_v003b"


def test_registering_does_not_change_active_model(tiny):
    # the tiny registry has no v001; just assert extra fields were stored and nothing is deployed
    entry = tiny["registry"].list_models()[0]
    assert entry["status"] == "pending_review" and entry["test_metrics"]["roc_auc"] == 0.65


# --------------------------------------------------------------------------- service
def test_service_returns_four_models_with_metrics_average_and_agreement(tiny):
    out = tiny["service"].compare(dict(VALID))
    assert [m["model_id"] for m in out["models"]] == list(MODEL_TAGS)
    probs = [m["probability"] for m in out["models"]]
    assert all(0 <= p <= 1 for p in probs)
    assert out["average_probability"] == pytest.approx(np.mean(probs), abs=1e-4)
    assert out["models_agree"] == (len({m["risk_level"] for m in out["models"]}) == 1)
    assert out["n_high"] + out["n_low"] == 4
    for m in out["models"]:
        assert m["risk_level"] == ("High" if m["probability"] >= m["threshold"] else "Low")
        assert 1 <= len(m["top_factors"]) <= 5
        assert m["test_metrics"]["roc_auc"] == 0.65 and m["test_metrics"]["majority_baseline_accuracy"] == pytest.approx(0.9003)
    assert "not the accuracy of the prediction for this patient" in out["metrics_note"]


def test_service_returns_both_operating_points_and_defaults_to_accuracy(tiny):
    out = tiny["service"].compare(dict(VALID))
    assert out["operating_point"] == "accuracy"
    for m in out["models"]:
        assert m["operating_point"] == "accuracy" and m["threshold"] == OP_ACC["threshold"]
        assert set(m["operating_points"]) == {"accuracy", "balanced"}
        for pt, op in (("accuracy", OP_ACC), ("balanced", OP_BAL)):
            got = m["operating_points"][pt]
            assert got["threshold"] == op["threshold"]
            assert got["risk_level"] == ("High" if m["probability"] >= op["threshold"] else "Low")
            for k in ("accuracy", "sensitivity", "specificity", "balanced_accuracy"):
                assert got["test"][k] == op[k]            # the page shows exactly the stored test metrics
    assert set(out["n_high_by_point"]) == {"accuracy", "balanced"}


def test_service_selected_operating_point_changes_headline_risk_level(tiny):
    acc = tiny["service"].compare(dict(VALID), operating_point="accuracy")
    bal = tiny["service"].compare(dict(VALID), operating_point="balanced")
    for a, b in zip(acc["models"], bal["models"]):
        assert a["probability"] == b["probability"]
        assert a["threshold"] == OP_ACC["threshold"] and b["threshold"] == OP_BAL["threshold"]
        assert b["risk_level"] == ("High" if b["probability"] >= OP_BAL["threshold"] else "Low")
        assert a["operating_points"] == b["operating_points"]
    assert bal["operating_point"] == "balanced"
    with pytest.raises(FeatureValidationError):
        tiny["service"].compare(dict(VALID), operating_point="nonsense")


def test_blank_inputs_are_not_reported_as_patient_factors(tiny):
    out = tiny["service"].compare({k: v for k, v in VALID.items() if k != "BPQ020"})   # BPQ020 + RIDRETH1 left blank
    for m in out["models"]:
        assert not {"BPQ020", "RIDRETH1"} & {f["feature"] for f in m["top_factors"]}
    assert any("Left blank" in w for w in out["warnings"])


def test_urdact_is_derived_and_optional_items_may_be_missing(tiny):
    row, warnings = tiny["service"].validate({k: v for k, v in VALID.items() if k != "BPQ020"})
    assert row["URDACT"] == pytest.approx(8.0 / 110.0 * 100.0)
    assert np.isnan(row["BPQ020"]) and np.isnan(row["RIDRETH1"])
    tiny["service"].compare({k: v for k, v in VALID.items() if k != "BPQ020"})   # still scores (imputed)


@pytest.mark.parametrize("mutation, expected", [
    ({"EXTRA": 1.0}, "Unknown feature"),
    ({"URDACT": 5.0}, "derived"),
    ({"RIDAGEYR": 5.0}, "between"),
    ({"BMXBMI": 500.0}, "between"),
    ({"BPQ020": 0.7}, "one of"),
    ({"RIDRETH1": 9.0}, "one of"),
    ({"RIDAGEYR": float("nan")}, "required"),
])
def test_service_rejects_invalid_inputs(tiny, mutation, expected):
    with pytest.raises(FeatureValidationError) as exc:
        tiny["service"].compare({**VALID, **mutation})
    assert expected in str(exc.value)


def test_service_requires_required_fields(tiny):
    with pytest.raises(FeatureValidationError) as exc:
        tiny["service"].compare({"RIDAGEYR": 50})
    assert "BMXBMI" in str(exc.value) and "URXUMA" in str(exc.value)


def test_service_reports_unavailable_models(tmp_path, tiny):
    empty = ModelRegistry(versions_dir=tmp_path / "v", manifest_path=tmp_path / "m.json")
    svc = NhanesCompareService(registry=empty, spec_path=tiny["service"]._spec_path)
    with pytest.raises(ModelsUnavailableError):
        svc.compare(dict(VALID))


# --------------------------------------------------------------------------- HTTP endpoint
@pytest.fixture
def patched(monkeypatch, tiny):
    monkeypatch.setattr("app.api.v1.routes.predict_router.nhanes_compare_service", tiny["service"])
    return tiny["service"]


def test_endpoint_requires_authentication(client, patched):
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID})
    assert res.status_code in (401, 403)


def test_endpoint_returns_four_models(client, auth_headers_hospital_1, patched):
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID, "hospital_id": 1}, headers=auth_headers_hospital_1)
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["models"]) == 4
    assert {m["model_family"] for m in body["models"]} == {"ebm", "catboost", "xgboost", "lightgbm"}
    assert isinstance(body["models_agree"], bool) and 0 <= body["average_probability"] <= 1
    assert body["models"][0]["test_metrics"]["majority_baseline_accuracy"] == pytest.approx(0.9003)
    assert "Research prototype" in body["disclaimer"]


def test_endpoint_returns_both_operating_points(client, auth_headers_hospital_1, patched):
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID, "hospital_id": 1}, headers=auth_headers_hospital_1)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["operating_point"] == "accuracy" and set(body["n_high_by_point"]) == {"accuracy", "balanced"}
    for m in body["models"]:
        assert set(m["operating_points"]) == {"accuracy", "balanced"}
        assert m["operating_points"]["accuracy"]["test"]["accuracy"] == OP_ACC["accuracy"]
        assert m["operating_points"]["balanced"]["test"]["sensitivity"] == OP_BAL["sensitivity"]
        assert m["operating_points"]["accuracy"]["risk_level"] in ("Low", "High")
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID, "operating_point": "balanced"}, headers=auth_headers_hospital_1)
    assert res.status_code == 200 and res.json()["operating_point"] == "balanced"
    assert all(m["threshold"] == OP_BAL["threshold"] for m in res.json()["models"])
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID, "operating_point": "x"}, headers=auth_headers_hospital_1)
    assert res.status_code == 422


def test_endpoint_validates_inputs(client, auth_headers_hospital_1, patched):
    res = client.post("/api/v1/predict/risk/compare", json={"features": {**VALID, "RIDAGEYR": 500}}, headers=auth_headers_hospital_1)
    assert res.status_code == 422
    assert any("between" in p for p in res.json()["detail"]["problems"])
    res = client.post("/api/v1/predict/risk/compare", json={"features": {**VALID, "SEQN": 1}}, headers=auth_headers_hospital_1)
    assert res.status_code == 422
    res = client.post("/api/v1/predict/risk/compare", json={}, headers=auth_headers_hospital_1)
    assert res.status_code == 422


def test_endpoint_enforces_tenant_scoping(client, db, auth_headers_hospital_2, patched):
    from app.db.models import Patient

    other = db.query(Patient).filter(Patient.hospital_id == 1).first()
    if other is None:
        pytest.skip("no hospital-1 patient seeded")
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID, "hospital_id": 1, "patient_id": other.id},
                      headers=auth_headers_hospital_2)
    assert res.status_code == 404   # a hospital-2 user cannot score against a hospital-1 patient


def test_endpoint_does_not_store_features(client, db, auth_headers_hospital_1, patched):
    from app.db.models import Prediction

    before = db.query(Prediction).count()
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID}, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    db.expire_all()
    assert db.query(Prediction).count() == before


def test_endpoint_503_when_models_missing(client, auth_headers_hospital_1, monkeypatch, tmp_path, tiny):
    empty = ModelRegistry(versions_dir=tmp_path / "v", manifest_path=tmp_path / "m.json")
    svc = NhanesCompareService(registry=empty, spec_path=tiny["service"]._spec_path)
    monkeypatch.setattr("app.api.v1.routes.predict_router.nhanes_compare_service", svc)
    res = client.post("/api/v1/predict/risk/compare", json={"features": VALID}, headers=auth_headers_hospital_1)
    assert res.status_code == 503


def test_spec_endpoint(client, auth_headers_hospital_1, patched):
    res = client.get("/api/v1/predict/risk/compare/spec", headers=auth_headers_hospital_1)
    assert res.status_code == 200
    names = {f["name"] for f in res.json()["features"]}
    assert "URDACT" in names and "KIQ029" not in names


def test_existing_urine_risk_endpoint_still_works(client, auth_headers_hospital_1):
    payload = {"age": 42, "gender": "male", "bmi": 27.4, "blood_pressure": 132, "diabetes": False, "family_history": False,
               "water_intake": 2.1, "urine_ph": 6.1, "urine_specific_gravity": 1.02, "calcium": 24, "uric_acid": 6.5,
               "creatinine": 0.9, "osmolality": 550, "conductivity": 22, "urea": 250}
    res = client.post("/api/v1/predict/risk", json=payload, headers=auth_headers_hospital_1)
    assert res.status_code == 200
    assert res.json()["risk_level"] in ("Low", "High")


# --------------------------------------------------------------------------- real registered models
def _real_models_registered() -> bool:
    from ml.risk_models import registry as real

    tags = {e.get("version_tag") for e in real.list_models()}
    return set(MODEL_TAGS) <= tags and nc.PROJECT_ROOT.joinpath("ml/artifacts/nhanes_v003/feature_spec.json").exists()


@pytest.mark.skipif(not _real_models_registered(), reason="nhanes_v003b models not registered (run ml/training/run_nhanes_notebooks.py)")
class TestRealRegisteredModels:
    def test_real_models_score_the_form_defaults(self):
        from app.services.nhanes_compare_service import nhanes_compare_service as svc

        spec = svc.get_spec()
        defaults = {f["name"]: f["default"] for f in spec["features"] if f["type"] != "derived" and f["default"] is not None}
        out = svc.compare(defaults)
        assert len(out["models"]) == 4
        for m in out["models"]:
            assert 0 < m["threshold"] < 1 and m["status"] == "pending_review"
            assert m["test_metrics"]["test_cycle"] == "2017-2018"
            assert set(m["operating_points"]) == {"accuracy", "balanced"}

    def test_page_numbers_equal_metrics_json(self):
        """What the compare endpoint (and so the page) shows must equal the saved metrics JSON of each model."""
        from app.services.nhanes_compare_service import nhanes_compare_service as svc

        spec = svc.get_spec()
        defaults = {f["name"]: f["default"] for f in spec["features"] if f["type"] != "derived" and f["default"] is not None}
        out = svc.compare(defaults)
        for m in out["models"]:
            key = m["model_family"]
            j = nc.load_json(nc.PROJECT_ROOT / "ml" / "outputs" / "metrics" / "nhanes_v003" / f"{key}.json")
            t = j["variants"][j["primary_variant"]]["test"]
            for pt in ("accuracy", "balanced"):
                shown, saved = m["operating_points"][pt], t["at_threshold"][pt]
                assert shown["threshold"] == pytest.approx(saved["threshold"], abs=1e-4)
                for k in ("accuracy", "sensitivity", "specificity", "balanced_accuracy"):
                    assert shown["test"][k] == pytest.approx(saved[k], abs=1e-9)
            assert m["test_metrics"]["roc_auc"] == pytest.approx(t["roc_auc"], abs=1e-9)
            assert m["test_metrics"]["majority_baseline_accuracy"] == pytest.approx(t["majority_baseline_accuracy"], abs=1e-9)

    def test_family_lookup_still_returns_legacy_wrapper_for_old_callers(self):
        from ml.risk_models import registry as real

        assert type(real.get_model_by_family("xgboost")) is XGBoostRiskModel   # not the v003 bundle wrapper

    def test_real_models_exclude_banned_columns_and_stay_undeployed(self):
        from ml.risk_models import registry as real

        for tag in MODEL_TAGS:
            model = real.get_model(tag)
            assert not set(model.bundle.input_features) & set(BANNED_COLUMNS)
        entries = {e["version_tag"]: e for e in real.list_models()}
        assert all(not entries[t]["is_deployed"] for t in MODEL_TAGS)
        assert real._manifest["active_model_version"] == "logistic_regression_v001"
        assert "xgboost_nhanes_v002" in entries and "logistic_regression_v001" in entries
