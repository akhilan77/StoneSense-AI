"""Tests for the nhanes_v003b revision: class-imbalance handling, the two operating points, the LightGBM early-stop fix,
the metric-table helper and the frontend build.

Unit tests use synthetic data. Tests marked "real artifacts" read the saved outputs of the executed notebooks and fail
loudly (not skip) when those outputs are missing, because the notebooks are part of the deliverable.
"""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import nhanes_common as nc  # noqa: E402

METRICS_DIR = PROJECT_ROOT / "ml" / "outputs" / "metrics" / nc.MODEL_VERSION
MODELS = ["ebm", "catboost", "xgboost", "lightgbm"]


def _real_metrics(key: str) -> dict:
    path = METRICS_DIR / f"{key}.json"
    assert path.exists(), f"{path.name} missing - execute the notebooks with ml/training/run_nhanes_notebooks.py"
    return nc.load_json(path)


# ---------------------------------------------------------------------------------------------- resampling is train-only
class _Spy(ClassifierMixin, BaseEstimator):
    """Minimal estimator recording what the Pipeline hands to ``fit``."""

    def fit(self, X, y, eval_set=None):
        self.fit_shape = X.shape
        self.fit_pos_rate = float(np.mean(y))
        self.eval_set_seen = eval_set
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.tile([1 - self.p_, self.p_], (len(X), 1))


def _imbalanced(n=3000, rate=0.1, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 6)
    X[:, 4] = rng.randint(0, 2, n)           # a binary column (sex-like)
    X[:, 5] = rng.randint(0, 2, n)           # a binary column (history-like)
    y = (rng.rand(n) < rate).astype(int)
    return X.astype(np.float32), y


@pytest.mark.parametrize("strategy", ["none", "class_weight", "random_over", "random_under", "smotenc"])
def test_resampling_touches_training_rows_only(strategy):
    Xtr, ytr = _imbalanced(3000, 0.1, 0)
    Xva, yva = _imbalanced(1000, 0.106, 1)
    Xte, yte = _imbalanced(1000, 0.10, 2)
    Xva0, yva0, Xte0, yte0 = Xva.copy(), yva.copy(), Xte.copy(), yte.copy()
    spy = _Spy()
    pipe = nc.fit_with_strategy(spy, strategy, Xtr, ytr, cat_idx=[4, 5], fit_params={"eval_set": [(Xva, yva)]})
    clf = pipe.named_steps["clf"]
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    expected_rows = {"none": len(ytr), "class_weight": len(ytr), "random_over": 2 * n_neg, "random_under": 2 * n_pos, "smotenc": 2 * n_neg}[strategy]
    assert clf.fit_shape[0] == expected_rows                      # the estimator saw the resampled TRAINING rows
    if strategy in ("random_over", "random_under", "smotenc"):
        assert clf.fit_pos_rate == pytest.approx(0.5, abs=1e-6)   # ...balanced 1:1
    else:
        assert clf.fit_pos_rate == pytest.approx(ytr.mean())
    # validation rows handed in as the eval set are untouched, and so are validation / test arrays and their prevalence
    assert clf.eval_set_seen[0][0] is Xva and clf.eval_set_seen[0][1] is yva
    np.testing.assert_array_equal(Xva, Xva0)
    np.testing.assert_array_equal(yva, yva0)
    np.testing.assert_array_equal(Xte, Xte0)
    assert yva.mean() == yva0.mean() and yte.mean() == yte0.mean()
    # predicting does not resample: one score per validation / test row
    assert len(pipe.predict_proba(Xva)) == len(yva) and len(pipe.predict_proba(Xte)) == len(yte)
    assert len(ytr) == 3000 and ytr.mean() == pytest.approx(_imbalanced(3000, 0.1, 0)[1].mean())   # the caller's training labels are not mutated


def test_smotenc_keeps_binary_columns_valid():
    Xtr, ytr = _imbalanced(3000, 0.1, 0)
    sampler = nc.make_sampler("smotenc", [4, 5])
    Xr, yr = sampler.fit_resample(Xtr, ytr)
    assert set(np.unique(Xr[:, 4])) <= {0.0, 1.0} and set(np.unique(Xr[:, 5])) <= {0.0, 1.0}   # SMOTE would create 0.37 etc.
    assert len(yr) > len(ytr)


def test_categorical_column_indices_cover_binary_and_onehot_columns():
    names = ["RIDAGEYR", "RIAGENDR", "BMXBMI", "BPQ020", "DIQ010", "INDFMPIR", "RIDRETH1_1.0", "RIDRETH1_2.0"]
    assert nc.categorical_column_indices(names) == [1, 3, 4, 6, 7]


def test_split_validation_is_stratified_disjoint_and_deterministic():
    rng = np.random.RandomState(0)
    y = (rng.rand(2000) < 0.1).astype(int)
    seqn = np.arange(1000, 3000)
    a, b = nc.split_validation(seqn, y), nc.split_validation(seqn, y)
    assert not set(a["es"]) & set(a["cal"]) and len(a["es"]) + len(a["cal"]) == 2000
    assert abs(y[a["es"]].mean() - y[a["cal"]].mean()) < 0.01
    np.testing.assert_array_equal(a["es"], b["es"])
    assert a["hash"] == b["hash"]


def test_select_strategy_rule_prefers_none_when_tied():
    tab = pd.DataFrame({"oof_pr_auc": [0.190, 0.191, 0.192, 0.180, 0.170], "oof_brier": [0.0917, 0.0918, 0.0919, 0.092, 0.0929]},
                       index=nc.IMBALANCE_STRATEGIES)
    pick, info = nc.select_strategy(tab)
    assert pick == "none" and info["none_within_tie_band"] and info["pr_auc_gain_over_none"] == 0.0
    tab.loc["smotenc", "oof_pr_auc"] = 0.25
    assert nc.select_strategy(tab)[0] == "smotenc"


# ---------------------------------------------------------------------------------------------- operating points
def test_operating_points_use_only_the_scores_they_are_given():
    rng = np.random.RandomState(1)
    yv, pv = (rng.rand(1500) < 0.1).astype(int), np.clip(0.1 + 0.3 * rng.rand(1500) + 0.1 * (rng.rand(1500) < 0.1), 0, 1)
    before = nc.operating_points(yv, pv)
    _unrelated_test_scores = (rng.rand(800) < 0.5).astype(int), rng.rand(800)    # never passed in
    after = nc.operating_points(yv, pv)
    assert before[nc.OP_BALANCED]["threshold"] == after[nc.OP_BALANCED]["threshold"]
    assert before[nc.OP_ACCURACY]["threshold"] == after[nc.OP_ACCURACY]["threshold"]
    assert set(before) == {nc.OP_BALANCED, nc.OP_ACCURACY}


def test_accuracy_point_maximises_sensitivity_subject_to_72_percent():
    rng = np.random.RandomState(4)
    y = (rng.rand(5000) < 0.1).astype(int)
    p = np.clip(0.08 + 0.12 * y + rng.randn(5000) * 0.07, 0.001, 0.999)
    got = nc.threshold_for_accuracy(y, p, nc.ACC_TARGET_VALIDATION)
    assert got is not None and got["accuracy"] >= nc.ACC_TARGET_VALIDATION
    qualifying = [nc.metrics_at_threshold(y, p, t) for t in np.unique(p) if nc.metrics_at_threshold(y, p, t)["accuracy"] >= nc.ACC_TARGET_VALIDATION]
    assert got["sensitivity"] == pytest.approx(max(q["sensitivity"] for q in qualifying))
    assert nc.threshold_for_accuracy(y, p, 0.999) is None                      # reported as unreachable, never faked


def test_extended_metrics_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0])
    m = nc.metrics_at_threshold(y, p, 0.5)       # TP=2 FP=1 FN=1 TN=6
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["positive_rate"] == pytest.approx(0.3)
    mcc = (2 * 6 - 1 * 1) / np.sqrt(3 * 3 * 7 * 7)
    assert m["mcc"] == pytest.approx(mcc)


def test_show_metrics_table_has_every_required_column_and_refuses_ci_without_frozen_thresholds(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nc, "metrics_dir", lambda: tmp_path)
    rng = np.random.RandomState(0)
    y = (rng.rand(600) < 0.1).astype(int)
    p = np.clip(0.1 + 0.2 * y + rng.rand(600) * 0.2, 0, 1)
    df = nc.show_metrics_table({"a": (y, p)}, "unit test", name="unit")
    for col in ["ROC-AUC", "PR-AUC", "Brier", "Accuracy", "Balanced acc", "Sensitivity", "Specificity", "Precision", "F1", "MCC",
                "Majority acc", "Pos-pred rate", "TN", "FP", "FN", "TP"]:
        assert col in df.columns
    out = capsys.readouterr().out
    assert "METRICS TABLE: unit test" in out and "Accuracy" in out
    assert (tmp_path / "tables" / "unit.csv").exists()
    assert float(df.iloc[0]["Majority acc"]) == pytest.approx(1 - y.mean(), abs=1e-3)
    with pytest.raises(ValueError):
        nc.show_metrics_table({"a": (y, p)}, "ci", ci=True)
    ci = nc.show_metrics_table({"a": (y, p)}, "ci", thresholds={"a": {"balanced": 0.2, "accuracy": 0.25}}, ci=True, quiet=True)
    assert "[" in ci.iloc[0]["Accuracy"]


def test_run_script_only_executes_notebooks():
    tree = ast.parse((PROJECT_ROOT / "ml" / "training" / "run_nhanes_notebooks.py").read_text(encoding="utf-8"))
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | \
               {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"sklearn", "lightgbm", "xgboost", "catboost", "interpret", "numpy", "pandas", "imblearn"}


# ---------------------------------------------------------------------------------------------- real artifacts
@pytest.mark.parametrize("key", MODELS)
def test_real_accuracy_point_rule_holds_on_validation(key):
    j = _real_metrics(key)
    for v, r in j["variants"].items():
        a = r["thresholds"][nc.OP_ACCURACY]
        if a is None:
            continue                                # reported as unreachable; never faked
        assert a["accuracy"] >= nc.ACC_TARGET_VALIDATION - 1e-9, (key, v)
        b = r["thresholds"][nc.OP_BALANCED]
        assert b["balanced_accuracy"] >= a["balanced_accuracy"] - 1e-9   # the balanced point really maximises balanced accuracy


@pytest.mark.parametrize("key", MODELS)
def test_real_saved_thresholds_reproduce_from_validation_half_only(key):
    """Re-derive both thresholds from the frozen bundle's calibrated scores on the validation calibration half; they must
    equal the stored ones. (Nothing in this check can see the test cycle.)"""
    import joblib

    j = _real_metrics(key)
    v = j["primary_variant"]
    splits = nc.load_frozen_splits(include_test=False)
    bundle = joblib.load(j["variants"][v]["bundle_path"])
    val = splits["validation"]
    half = nc.split_validation(val["SEQN"].values, val["target_stone"].values)
    cal_rows = val.iloc[half["cal"]]
    p = bundle.predict_proba(cal_rows)
    ops = nc.operating_points(cal_rows["target_stone"].values, p)
    for pt in nc.OPERATING_POINTS:
        stored = bundle.thresholds[pt]
        if ops[pt] is None:
            assert stored is None
        else:
            # XGBoost is trained on the GPU and served on the CPU, so scores agree to ~1e-6, not bit for bit
            assert stored == pytest.approx(ops[pt]["threshold"], abs=1e-4)
            if pt == nc.OP_ACCURACY:
                assert nc.metrics_at_threshold(cal_rows["target_stone"].values, p, stored)["accuracy"] >= nc.ACC_TARGET_VALIDATION - 1e-3


def test_real_lightgbm_does_not_stop_early():
    j = _real_metrics("lightgbm")
    t = j["variants"][j["primary_variant"]]["tuning"]
    assert t["best_iteration"] > 30, t
    assert j["diagnostics"]["lightgbm_early_stop"]["first_metric_only_new"] is True


@pytest.mark.parametrize("key", ["xgboost", "catboost"])
def test_real_other_boosters_do_not_stop_suspiciously_early(key):
    j = _real_metrics(key)
    assert j["variants"][j["primary_variant"]]["tuning"]["best_iteration"] > 30


def test_real_registry_has_both_operating_points_with_validation_and_test_metrics():
    manifest = nc.load_json(PROJECT_ROOT / "ml" / "models" / "registry_manifest.json")
    for k in MODELS:
        entry = manifest["models"][f"{k}_{nc.RUN_TAG}"]
        assert entry["status"] == "pending_review" and entry["is_deployed"] is False
        assert entry["default_operating_point"] == nc.OP_ACCURACY
        j = _real_metrics(k)
        saved = j["variants"][j["primary_variant"]]["test"]["at_threshold"]
        for pt in nc.OPERATING_POINTS:
            op = entry["operating_points"][pt]
            assert op["validation"] is not None and op["test"] is not None
            assert op["test"]["accuracy"] == pytest.approx(saved[pt]["accuracy"], abs=1e-12)
            assert op["test"]["sensitivity"] == pytest.approx(saved[pt]["sensitivity"], abs=1e-12)
            assert op["threshold"] == pytest.approx(saved[pt]["threshold"], abs=1e-12)
    for k in MODELS:   # earlier versions are kept
        assert f"{k}_nhanes_v003" in manifest["models"]
    assert manifest["active_model_version"] == "logistic_regression_v001"


# ---------------------------------------------------------------------------------------------- frontend
FRONTEND = PROJECT_ROOT / "frontend"


def _run_npx(args, tmp_dir=None):
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx is None or not (FRONTEND / "node_modules").exists():
        pytest.skip("node / frontend dependencies not installed")
    return subprocess.run([npx, *args], cwd=FRONTEND, capture_output=True, text=True, timeout=600)


def test_frontend_type_checks():
    res = _run_npx(["tsc", "--noEmit"])
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-1000:]


def test_frontend_builds(tmp_path):
    res = _run_npx(["vite", "build", "--outDir", str(tmp_path / "dist"), "--emptyOutDir"])
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-1000:]
    assert (tmp_path / "dist" / "index.html").exists()
