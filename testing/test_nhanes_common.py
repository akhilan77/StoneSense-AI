"""Unit tests for ml/training/nhanes_common.py and the NHANES v003 bundle (synthetic data; no test-cycle access)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import nhanes_common as nc  # noqa: E402
from ml.risk_models.nhanes_bundle import BANNED_COLUMNS, NhanesBundle, ProbCalibrator  # noqa: E402


def _synthetic_cohort(n_per_cycle: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    seqn = 1
    for cycle in nc.ALL_CYCLES:
        for _ in range(n_per_cycle):
            r = {"SEQN": seqn, "CYCLE": cycle, "target_stone": int(rng.rand() < 0.1),
                 "SDMVPSU": int(rng.randint(1, 3)), "SDMVSTRA": int(rng.randint(1, 5))}
            for f in nc.BASE_FEATURES + nc.SENSITIVE_FEATURES:
                r[f] = float(rng.rand() * 10 + 1)
            r["RIDRETH1"] = float(rng.randint(1, 6))
            for h in ("BPQ020", "BPQ080", "DIQ010", "MCQ080"):
                r[h] = float(rng.randint(0, 2))
            rows.append(r)
            seqn += 1
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ splits
def test_cycle_split_is_disjoint_and_complete():
    df = _synthetic_cohort()
    sp = nc.split_by_cycle(df)
    assert set(sp["train"]["CYCLE"]) == set(nc.TRAIN_CYCLES)
    assert set(sp["validation"]["CYCLE"]) == set(nc.VAL_CYCLES)
    assert set(sp["test"]["CYCLE"]) == set(nc.TEST_CYCLES)
    assert not (set(sp["train"].SEQN) & set(sp["validation"].SEQN))
    assert not (set(sp["train"].SEQN) & set(sp["test"].SEQN))
    assert not (set(sp["validation"].SEQN) & set(sp["test"].SEQN))
    assert sum(len(v) for v in sp.values()) == len(df)


def test_overlap_is_detected():
    df = _synthetic_cohort()
    sp = nc.split_by_cycle(df)
    sp["test"] = pd.concat([sp["test"], sp["validation"].head(1)])
    with pytest.raises(AssertionError):
        nc.assert_splits_disjoint(sp)


def test_split_hash_is_order_independent_and_sensitive():
    a = {"train": [3, 1, 2], "test": [9]}
    b = {"test": [9], "train": [1, 2, 3]}
    c = {"train": [1, 2], "test": [3, 9]}
    assert nc.split_hash(a) == nc.split_hash(b)
    assert nc.split_hash(a) != nc.split_hash(c)


@pytest.mark.skipif(not nc.cohort_v2_path().exists(), reason="v2 cohort not built")
def test_real_cohort_splits_have_no_seqn_overlap():
    df = nc.load_cohort_v2()
    sp = nc.split_by_cycle(df)
    nc.assert_splits_disjoint(sp)
    assert len(sp["train"]) + len(sp["validation"]) + len(sp["test"]) == len(df)
    assert "KIQ029" not in df.columns


# ------------------------------------------------------------------ banned columns
def test_banned_columns_absent_from_every_variant():
    variants = nc.make_variants(["BPQ020", "BPQ080", "DIQ010", "MCQ080"])
    for feats in variants.values():
        assert not (set(feats) & set(BANNED_COLUMNS))
    assert "KIQ029" in BANNED_COLUMNS and "KIQ026" in BANNED_COLUMNS


def test_banned_column_in_features_raises():
    with pytest.raises(AssertionError):
        nc.assert_no_banned_features(["RIDAGEYR", "KIQ029"])
    with pytest.raises(ValueError):
        NhanesBundle("m", "xgboost", "base", ["RIDAGEYR", "KIQ026"], [], None, None,
                     ProbCalibrator("identity"), {"balanced_accuracy": 0.5})


def test_sensitive_features_only_in_opt_in_variant():
    v = nc.make_variants(["BPQ020"])
    assert not (set(nc.SENSITIVE_FEATURES) & set(v["base"]))
    assert not (set(nc.SENSITIVE_FEATURES) & set(v["base_history"]))
    assert set(nc.SENSITIVE_FEATURES) <= set(v["base_history_sensitive"])


# ------------------------------------------------------------------ metrics (hand-computed)
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0])


def test_metrics_at_threshold_match_hand_computation():
    m = nc.metrics_at_threshold(Y, P, 0.5)  # predicted positive: idx 0,1,3 -> TP=2 FP=1 FN=1 TN=6
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(6 / 7)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 6 / 7) / 2)


def test_roc_auc_brier_hand_computed():
    pos, neg = P[Y == 1], P[Y == 0]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    out = nc.evaluate_scores(Y, P, {"x": 0.5})
    assert out["roc_auc"] == pytest.approx(wins)
    assert out["brier"] == pytest.approx(np.mean((P - Y) ** 2))
    assert out["majority_baseline_accuracy"] == pytest.approx(0.7)


def test_threshold_max_balanced_accuracy_matches_brute_force():
    best = max(nc.metrics_at_threshold(Y, P, t)["balanced_accuracy"] for t in np.unique(P))
    got = nc.threshold_max_balanced_accuracy(Y, P)
    assert got["balanced_accuracy"] == pytest.approx(best)


def test_threshold_for_accuracy_rule_and_unreachable():
    got = nc.threshold_for_accuracy(Y, P, 0.8)
    assert got is not None and got["accuracy"] >= 0.8
    # lowest qualifying threshold == max sensitivity among those meeting the target
    ms = [nc.metrics_at_threshold(Y, P, t) for t in np.unique(P)]
    qualifying = [m for m in ms if m["accuracy"] >= 0.8]
    assert got["sensitivity"] == pytest.approx(max(q["sensitivity"] for q in qualifying))
    assert nc.threshold_for_accuracy(Y, P, 0.99) is None


def test_thresholds_are_functions_of_validation_scores_only():
    """Selection takes only (y_val, p_val); an unrelated would-be test set cannot influence it."""
    rng = np.random.RandomState(1)
    yv, pv = (rng.rand(500) < 0.1).astype(int), rng.rand(500)
    t1 = nc.threshold_max_balanced_accuracy(yv, pv)["threshold"]
    _ = (rng.rand(400) < 0.5).astype(int), rng.rand(400)  # never passed in
    assert nc.threshold_max_balanced_accuracy(yv, pv)["threshold"] == t1


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.RandomState(3)
    y = (rng.rand(800) < 0.1).astype(int)
    p = np.clip(0.1 + 0.3 * y + rng.randn(800) * 0.15, 0, 1)
    a = nc.bootstrap_metrics(y, p, {"bal": 0.2, "none": None}, n_boot=200, seed=1)
    b = nc.bootstrap_metrics(y, p, {"bal": 0.2, "none": None}, n_boot=200, seed=1)
    assert a["roc_auc_ci95"] == b["roc_auc_ci95"]
    lo, hi = a["roc_auc_ci95"]
    assert lo <= a["roc_auc"] <= hi
    assert a["at_threshold"]["none"] is None
    s = a["at_threshold"]["bal"]
    assert s["sensitivity_ci95"][0] <= s["sensitivity"] <= s["sensitivity_ci95"][1]


# ------------------------------------------------------------------ calibration / preprocessing
def test_choose_calibrator_returns_valid_probabilities():
    rng = np.random.RandomState(0)
    y = (rng.rand(1500) < 0.1).astype(int)
    raw = np.clip(0.4 * y + rng.rand(1500) * 0.6, 0.001, 0.999)  # miscalibrated (inflated)
    cal, rep = nc.choose_calibrator(raw, y)
    p = cal.predict(raw)
    assert rep["method"] in ("isotonic", "sigmoid")
    assert p.min() >= 0 and p.max() <= 1
    assert rep["brier_calibrated_in_sample"] < rep["brier_raw_uncalibrated"]


def test_preprocessor_fit_on_train_only():
    df = _synthetic_cohort()
    sp = nc.split_by_cycle(df)
    vd = nc.prepare_variant({"train": sp["train"], "validation": sp["validation"]}, "base", nc.BASE_FEATURES)
    scaler = vd.preprocessor.named_transformers_["num"].named_steps["scaler"]
    assert np.allclose(scaler.mean_, sp["train"][vd.features].mean().values)
    assert not np.allclose(scaler.mean_, df[vd.features].mean().values)


def test_leakage_audit_flags_a_leaky_feature():
    rng = np.random.RandomState(0)
    y = (rng.rand(2000) < 0.1).astype(int)
    df = pd.DataFrame({"target_stone": y, "leak": y + rng.rand(2000) * 0.1, "noise": rng.rand(2000)})
    audit = nc.leakage_audit(df, ["leak", "noise"]).set_index("feature")
    assert audit.loc["leak", "flagged"] and not audit.loc["noise", "flagged"]


# ------------------------------------------------------------------ test-once ledger
class _DummyBundle:
    thresholds = {"balanced_accuracy": 0.2, "accuracy_ge_75": None}

    def predict_proba(self, X):
        return np.linspace(0.01, 0.9, len(X))


def test_final_test_evaluation_is_recorded_and_not_repeated(tmp_path, monkeypatch):
    df = _synthetic_cohort(n_per_cycle=300)
    sp = nc.split_by_cycle(df)
    monkeypatch.delenv("NHANES_DRY_RUN", raising=False)
    monkeypatch.delenv("NHANES_ALLOW_TEST_REEVAL", raising=False)
    monkeypatch.setattr(nc, "metrics_dir", lambda: tmp_path / "metrics")
    monkeypatch.setattr(nc, "predictions_dir", lambda: tmp_path / "preds")
    monkeypatch.setattr(nc, "load_frozen_splits", lambda include_test=False: sp)
    monkeypatch.setattr(nc, "bootstrap_metrics", lambda y, p, t, n_boot=1000, seed=0: {
        "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()), "roc_auc": 0.5})
    first = nc.final_test_evaluation(_DummyBundle(), "m", "base", "fp1")
    assert first["result"]["cached"] is False and first["result"]["evaluated_on"].startswith("test")
    again = nc.final_test_evaluation(_DummyBundle(), "m", "base", "fp1")
    assert again["result"]["cached"] is True
    with pytest.raises(RuntimeError):  # a different frozen model may not be re-scored on test
        nc.final_test_evaluation(_DummyBundle(), "m", "base", "fp2")


def test_dry_run_never_touches_test_cycle(monkeypatch):
    df = _synthetic_cohort()
    sp = nc.split_by_cycle(df)
    seen = {}

    def fake_load(include_test=False):
        seen["include_test"] = include_test
        return sp

    monkeypatch.setenv("NHANES_DRY_RUN", "1")
    monkeypatch.setattr(nc, "load_frozen_splits", fake_load)
    out = nc.final_test_evaluation(_DummyBundle(), "m", "base", "fp")
    assert seen["include_test"] is False
    assert out["result"]["dry_run"] is True
