"""Tests on the EXECUTED notebooks (ml/notebooks/nhanes_v003/*.ipynb): saved outputs, errors, provenance.

The notebooks are the deliverable, so a notebook without real outputs fails these tests (it does not skip).
"""
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import nhanes_common as nc  # noqa: E402

NB_DIR = PROJECT_ROOT / "ml" / "notebooks" / "nhanes_v003"
METRICS_DIR = PROJECT_ROOT / "ml" / "outputs" / "metrics" / nc.MODEL_VERSION
MODEL_NB = {"ebm": "01_ebm.ipynb", "catboost": "02_catboost.ipynb", "xgboost": "03_xgboost.ipynb", "lightgbm": "04_lightgbm.ipynb",
            "logreg": "04b_logistic_regression_baseline.ipynb"}
ALL_NB = ["00_data_and_splits.ipynb", *MODEL_NB.values(), "05_compare_and_register.ipynb"]


def _load(name):
    return json.loads((NB_DIR / name).read_text(encoding="utf-8"))


def _cell_text(cell):
    out = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            text = "".join(o["text"])
            if out and out[-1][0] == "stream":          # a stream may arrive in several chunks: concatenate them as printed
                out[-1] = ("stream", out[-1][1] + text)
            else:
                out.append(("stream", text))
        elif o["output_type"] in ("display_data", "execute_result") and "text/plain" in o["data"]:
            out.append(("display", "".join(o["data"]["text/plain"])))
    return "\n".join(t for _, t in out)


def _all_output(nb):
    return "\n".join(_cell_text(c) for c in nb["cells"] if c["cell_type"] == "code")


def _tables(nb):
    """{title: text block} for every 'METRICS TABLE' block printed by nhanes_common.show_metrics_table."""
    found = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        text = _cell_text(c)
        for m in re.finditer(r"=== METRICS TABLE: (.+?) ===\n((?:.*\n?)*?)(?=\n\s*\n|\Z|=== )", text):
            found[m.group(1)] = m.group(2)
    return found


@pytest.mark.parametrize("name", ALL_NB)
def test_notebook_was_executed_without_errors(name):
    nb = _load(name)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c.get("execution_count") for c in code), f"{name}: not every code cell was executed top to bottom"
    for c in code:
        assert not any(o["output_type"] == "error" for o in c.get("outputs", [])), f"{name}: error output in a code cell"
    counts = [c["execution_count"] for c in code]
    assert counts == sorted(counts) and counts[0] == 1, f"{name}: cells were not run in order in a fresh kernel"


@pytest.mark.parametrize("key", list(MODEL_NB))
def test_model_notebook_shows_the_full_metric_table_at_every_stage(key):
    nb = _load(MODEL_NB[key])
    tables = _tables(nb)
    titles = list(tables)
    assert any(t.startswith("BASELINE") for t in titles), titles
    assert any(t.startswith("AFTER TUNING") for t in titles), titles
    assert any(t.startswith("AFTER CALIBRATION") for t in titles), titles
    assert any(t.startswith("FINAL TEST EVALUATION") for t in titles), titles
    val_strat = next(t for t in titles if t.startswith("IMBALANCE STRATEGIES") and t.endswith("- validation"))
    test_strat = next(t for t in titles if t.startswith("IMBALANCE STRATEGIES") and "held-out TEST" in t)
    columns = ["Accuracy", "Balanced acc", "Sensitivity", "Specificity", "Precision", "F1", "MCC", "ROC-AUC", "PR-AUC", "Brier",
               "Majority acc", "Pos-pred rate", "TN", "FP", "FN", "TP"]
    for title in [next(t for t in titles if t.startswith(p)) for p in ("BASELINE", "AFTER TUNING", "AFTER CALIBRATION", "FINAL TEST EVALUATION")] + [val_strat, test_strat]:
        header = tables[title].splitlines()[0]
        for col in columns:
            assert col in header, (key, title, col)
    for s in nc.IMBALANCE_STRATEGIES:
        assert re.search(rf"^\s*{s}\s+(balanced|accuracy)\s", tables[val_strat], re.M), (key, "validation", s)
        assert re.search(rf"^\s*{s}\s+(balanced|accuracy)\s", tables[test_strat], re.M), (key, "test", s)
    final = tables[next(t for t in titles if t.startswith("FINAL TEST EVALUATION"))]
    assert re.search(r"\baccuracy\b", final) and re.search(r"\bbalanced\b", final)       # both operating points
    assert "[" in final                                                                    # bootstrap CIs present
    out = _all_output(nb)
    assert "LEAVE-ONE-CYCLE-OUT CV, per fold" in out and "Accuracy (accuracy)" in out and "Sensitivity (balanced)" in out
    assert "SELECTED strategy" in out
    chart = nc.PROJECT_ROOT / "ml" / "outputs" / "charts" / nc.MODEL_VERSION / f"{key}_confusion_test.png"
    assert chart.exists()
    assert any("image/png" in o.get("data", {}) for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])), "no embedded plot output"


def test_lightgbm_notebook_shows_the_diagnosis():
    out = _all_output(_load(MODEL_NB["lightgbm"]))
    assert "verdict on the suspected causes" in out and "metric that triggered the old stop" in out
    assert "BEFORE best_iteration_" in out and "AFTER best_iteration_" in out
    assert "best_iteration_ and rounds trained, three boosted models" in out


def _provenance_from_output(nb, fallback_name=None):
    m = re.search(r"PROVENANCE notebook=(\S+) executed_at=(\S+) split_hash=(\S+)", _all_output(nb))
    assert m, "PROVENANCE line missing from the notebook output"
    return m.groups()


def _parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _exec_window(nb):
    stamps = []
    for c in nb["cells"]:
        ex = c.get("metadata", {}).get("execution", {})
        for k in ("iopub.execute_input", "shell.execute_reply"):
            if k in ex:
                stamps.append(datetime.fromisoformat(ex[k].replace("Z", "+00:00")))
    return (min(stamps), max(stamps)) if stamps else None


def test_provenance_matches_the_executed_notebooks_and_split_hash_is_identical():
    frozen = nc.load_json(PROJECT_ROOT / "ml" / "artifacts" / nc.MODEL_VERSION / "split_manifest.json")["hash"]
    data = nc.load_json(METRICS_DIR / "data_and_splits.json")
    assert data["split_manifest"]["hash"] == frozen
    hashes = set()
    for key, nb_name in MODEL_NB.items():
        j = nc.load_json(METRICS_DIR / f"{key}.json")
        p = j["provenance"]
        assert p["notebook"] == nb_name and j["model_name"] == f"{key}_{nc.RUN_TAG}"
        for field in ("executed_at", "device", "training_seconds", "libraries", "split_hash"):
            assert p.get(field) not in (None, "", {}), (key, field)
        assert "lightgbm" in p["libraries"] and "imblearn" in p["libraries"]
        nb = _load(nb_name)
        name_out, ts_out, hash_out = _provenance_from_output(nb)
        assert (name_out, ts_out, hash_out) == (nb_name, p["executed_at"], p["split_hash"]), key   # JSON == what the executed notebook printed
        window = _exec_window(nb)
        if window:                                                                                  # nbconvert records per-cell timing
            lo, hi = window
            assert lo.replace(microsecond=0) - timedelta(seconds=2) <= _parse(p["executed_at"]) <= hi + timedelta(seconds=2), key
        assert j["split_hash"] == p["split_hash"] == frozen
        hashes.add(p["split_hash"])
    assert hashes == {frozen}                                      # identical across all models


def test_notebook_05_loads_only_saved_artifacts_and_fails_loudly():
    nb = _load("05_compare_and_register.ipynb")
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "FileNotFoundError" in src and "split hash" in src and "provenance" in src
    for forbidden in (".fit(", "fit_with_strategy", "random_search", "final_test_evaluation(", "load_frozen_splits(include_test=True"):
        assert forbidden not in src, f"05 must not retrain / evaluate: found {forbidden}"
