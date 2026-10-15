"""Shared logic for the NHANES v003 model notebooks (ml/notebooks/nhanes_v003/).

Every notebook imports this module so all four models use identical data, preprocessing, calibration,
threshold rules, bootstrap metrics and artifact layout. Nothing here is model-specific.

Design rules enforced here
- Splits are by survey cycle and frozen on disk with a hash (``freeze_splits`` / ``load_frozen_splits``).
- The test cycle is only reachable through ``final_test_evaluation`` (one evaluation per frozen model,
  recorded in a ledger; a changed model is refused unless NHANES_ALLOW_TEST_REEVAL=1).
- Imputers / scalers are fit on the training cycles only. Class-imbalance resampling (random over/under, SMOTENC)
  is applied to TRAINING rows only, inside an imblearn Pipeline fit step, after the split; validation and test
  rows are never resampled.
- Thresholds, the calibration method and the imbalance strategy are chosen on the validation cycle only.

Set NHANES_DRY_RUN=1 to develop without touching the test cycle: all outputs go to ml/outputs/_dryrun/
and ``final_test_evaluation`` scores the *validation* cycle instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.risk_models.nhanes_bundle import BANNED_COLUMNS, NhanesBundle, ProbCalibrator  # noqa: E402

SEED = 42
MODEL_VERSION = "nhanes_v003"      # data / split / output directory name (unchanged, same frozen split)
RUN_TAG = "nhanes_v003b"           # registered model version tag (nhanes_v003 entries are kept alongside)
OP_BALANCED = "balanced"           # operating point: maximise balanced accuracy on validation
OP_ACCURACY = "accuracy"           # operating point: maximise sensitivity s.t. validation accuracy >= ACC_TARGET_VALIDATION
OPERATING_POINTS = (OP_BALANCED, OP_ACCURACY)
ACC_TARGET_VALIDATION = 0.72       # validation target (margin above the 70% goal for validation-to-test drift)
ACC_GOAL_TEST = 0.70               # the user-facing goal on the held-out test cycle
IMBALANCE_STRATEGIES = ["none", "class_weight", "random_over", "random_under", "smotenc"]
STRATEGY_LABEL = {"none": "none", "class_weight": "class weights", "random_over": "random oversampling",
                  "random_under": "random undersampling", "smotenc": "SMOTENC"}
STRATEGY_TIE_BAND = 0.005          # pre-declared: strategies within this PR-AUC of the best are treated as tied
TRAIN_CYCLES = ["2007-2008", "2009-2010", "2011-2012", "2013-2014"]
VAL_CYCLES = ["2015-2016"]
TEST_CYCLES = ["2017-2018"]
ALL_CYCLES = TRAIN_CYCLES + VAL_CYCLES + TEST_CYCLES

# 13 features of xgboost_nhanes_v002 + 4 additional clinical features
V002_FEATURES = [
    "RIDAGEYR", "RIAGENDR", "BMXBMI", "LBXSCA", "LBXSUA", "LBXSCR", "LBXSBU",
    "LBXSGL", "LBXSNASI", "LBXSKSI", "LBXSPH", "URXUCR", "URXUMA",
]
EXTRA_BASE_FEATURES = ["BMXWAIST", "LBXSCLSI", "LBXSTB", "URDACT"]
BASE_FEATURES = V002_FEATURES + EXTRA_BASE_FEATURES
SENSITIVE_FEATURES = ["RIDRETH1", "INDFMPIR"]  # race / income: ethically sensitive, opt-in only
CATEGORICAL_FEATURES = ["RIDRETH1"]
CANDIDATE_HISTORY = ["BPQ020", "BPQ080", "DIQ010", "MCQ080", "MCQ520", "MCQ550"]
VARIANT_NAMES = ["base", "base_history", "base_history_sensitive"]

MAJORITY_NOTE = (
    "About 90% of participants report no stone, so predicting 'no stone' for everyone already scores ~90% accuracy."
)

# --------------------------------------------------------------------------------------- feature metadata
FEATURE_META: Dict[str, Dict[str, Any]] = {
    "RIDAGEYR": dict(label="Age", unit="years", type="continuous", min=20, max=80, group="Demographics",
                     clinical_role="Stone incidence rises through mid-life"),
    "RIAGENDR": dict(label="Sex", unit="", type="binary", min=1, max=2, group="Demographics",
                     options=[{"value": 1, "label": "Male"}, {"value": 2, "label": "Female"}],
                     clinical_role="Stones are historically more prevalent in males"),
    "BMXBMI": dict(label="Body mass index", unit="kg/m2", type="continuous", min=10, max=90, group="Body measures",
                   clinical_role="Adiposity is associated with stone formation"),
    "BMXWAIST": dict(label="Waist circumference", unit="cm", type="continuous", min=40, max=200, group="Body measures",
                     clinical_role="Central adiposity / metabolic syndrome marker"),
    "LBXSCA": dict(label="Serum calcium", unit="mg/dL", type="continuous", min=5, max=16, group="Blood chemistry",
                   clinical_role="Calcium metabolism"),
    "LBXSUA": dict(label="Serum uric acid", unit="mg/dL", type="continuous", min=0.5, max=20, group="Blood chemistry",
                   clinical_role="Hyperuricemia predisposes to uric-acid stones"),
    "LBXSCR": dict(label="Serum creatinine", unit="mg/dL", type="continuous", min=0.1, max=20, group="Blood chemistry",
                   clinical_role="Kidney function"),
    "LBXSBU": dict(label="Blood urea nitrogen", unit="mg/dL", type="continuous", min=1, max=120, group="Blood chemistry",
                   clinical_role="Renal clearance / protein catabolism"),
    "LBXSGL": dict(label="Serum glucose", unit="mg/dL", type="continuous", min=20, max=800, group="Blood chemistry",
                   clinical_role="Glycaemic status"),
    "LBXSNASI": dict(label="Serum sodium", unit="mmol/L", type="continuous", min=100, max=170, group="Blood chemistry",
                     clinical_role="Electrolyte and volume balance"),
    "LBXSKSI": dict(label="Serum potassium", unit="mmol/L", type="continuous", min=1.5, max=8.5, group="Blood chemistry",
                    clinical_role="Electrolyte balance"),
    "LBXSPH": dict(label="Serum phosphorus", unit="mg/dL", type="continuous", min=0.5, max=12, group="Blood chemistry",
                   clinical_role="Mineral homeostasis"),
    "LBXSCLSI": dict(label="Serum chloride", unit="mmol/L", type="continuous", min=70, max=130, group="Blood chemistry",
                     clinical_role="Acid-base balance"),
    "LBXSTB": dict(label="Serum total bilirubin", unit="mg/dL", type="continuous", min=0, max=20, group="Blood chemistry",
                   clinical_role="Hepatic marker"),
    "URXUCR": dict(label="Urine creatinine", unit="mg/dL", type="continuous", min=1, max=900, group="Urine",
                   clinical_role="Urine concentration"),
    "URXUMA": dict(label="Urine albumin", unit="ug/mL", type="continuous", min=0.1, max=30000, group="Urine",
                   clinical_role="Glomerular permeability"),
    "URDACT": dict(label="Urine albumin/creatinine ratio", unit="mg/g", type="derived", min=0.1, max=100000, group="Urine",
                   derived_from=["URXUMA", "URXUCR"], formula="URXUMA / URXUCR * 100",
                   clinical_role="Derived from urine albumin and urine creatinine"),
    "RIDRETH1": dict(label="Race / ethnicity", unit="", type="categorical", min=1, max=5, group="Sensitive",
                     options=[{"value": 1, "label": "Mexican American"}, {"value": 2, "label": "Other Hispanic"},
                              {"value": 3, "label": "Non-Hispanic White"}, {"value": 4, "label": "Non-Hispanic Black"},
                              {"value": 5, "label": "Other / multiracial"}],
                     clinical_role="Sensitive demographic input; off by default"),
    "INDFMPIR": dict(label="Family income-to-poverty ratio", unit="ratio", type="continuous", min=0, max=5, group="Sensitive",
                     clinical_role="Sensitive socio-economic input; off by default"),
}
_YES_NO = [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]
HISTORY_META: Dict[str, Dict[str, Any]] = {
    "BPQ020": dict(label="Told you have high blood pressure", unit="", type="binary", min=0, max=1, group="Medical history",
                   options=_YES_NO),
    "BPQ080": dict(label="Told you have high cholesterol", unit="", type="binary", min=0, max=1, group="Medical history",
                   options=_YES_NO),
    "DIQ010": dict(label="Told you have diabetes", unit="", type="ordinal", min=0, max=1, group="Medical history",
                   options=[{"value": 0, "label": "No"}, {"value": 0.5, "label": "Borderline / pre-diabetes"},
                            {"value": 1, "label": "Yes"}]),
    "MCQ080": dict(label="Told you were overweight", unit="", type="binary", min=0, max=1, group="Medical history",
                   options=_YES_NO),
    "MCQ520": dict(label="Abdominal pain in the past 12 months", unit="", type="binary", min=0, max=1,
                   group="Medical history", options=_YES_NO),
    "MCQ550": dict(label="Told you have gallstones", unit="", type="binary", min=0, max=1, group="Medical history",
                   options=_YES_NO),
}


# --------------------------------------------------------------------------------------- paths / modes
def is_dry_run() -> bool:
    return os.environ.get("NHANES_DRY_RUN", "0") == "1"


def _dry_root() -> Path:
    return PROJECT_ROOT / "ml" / "outputs" / "_dryrun"


def datasets_dir() -> Path:
    return PROJECT_ROOT / "ml" / "datasets"


def cohort_v2_path() -> Path:
    return datasets_dir() / "nhanes_kidney_stone_cohort_v2.csv"


def splits_dir() -> Path:
    return (_dry_root() / "splits") if is_dry_run() else datasets_dir() / "nhanes_v003"


def artifacts_dir() -> Path:
    return (_dry_root() / "artifacts") if is_dry_run() else PROJECT_ROOT / "ml" / "artifacts" / MODEL_VERSION


def metrics_dir() -> Path:
    return (_dry_root() / "metrics") if is_dry_run() else PROJECT_ROOT / "ml" / "outputs" / "metrics" / MODEL_VERSION


def charts_dir() -> Path:
    return (_dry_root() / "charts") if is_dry_run() else PROJECT_ROOT / "ml" / "outputs" / "charts" / MODEL_VERSION


def reports_dir() -> Path:
    return (_dry_root() / "reports") if is_dry_run() else PROJECT_ROOT / "ml" / "outputs" / "reports"


def predictions_dir() -> Path:
    return (_dry_root() / "predictions") if is_dry_run() else PROJECT_ROOT / "ml" / "outputs" / "predictions" / MODEL_VERSION


def bundles_dir() -> Path:
    """Per-variant inference bundles written by the model notebooks (git-ignored)."""
    return (_dry_root() / "bundles") if is_dry_run() else PROJECT_ROOT / "ml" / "outputs" / "models" / MODEL_VERSION


def versions_dir() -> Path:
    return (_dry_root() / "versions") if is_dry_run() else PROJECT_ROOT / "ml" / "models" / "versions"


def manifest_path() -> Path:
    return (_dry_root() / "registry_manifest.json") if is_dry_run() else PROJECT_ROOT / "ml" / "models" / "registry_manifest.json"


def rel(path: Any) -> str:
    """Repo-relative POSIX path for printing (keeps local user names out of saved notebook outputs)."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_project_root(start: Optional[Path] = None) -> Path:
    """Walks up from ``start`` (default cwd) to the repo root; used by notebooks."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "ml" / "training" / "nhanes_common.py").exists():
            return cand
    raise FileNotFoundError("Could not locate the repo root from " + str(p))


def set_seeds(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: Path) -> Path:
    ensure(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2)
    return path


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------- devices
def probe_devices() -> Dict[str, Any]:
    """Reports GPU availability per library by actually fitting a tiny model; never raises."""
    info: Dict[str, Any] = {"nvidia_smi": None, "xgboost_cuda": False, "catboost_gpu": False, "lightgbm_gpu": False}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["nvidia_smi"] = out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    rng = np.random.RandomState(0)
    X = rng.rand(400, 4)
    y = (X[:, 0] + 0.1 * rng.rand(400) > 0.5).astype(int)
    try:
        import xgboost as xgb

        xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=5, verbosity=0).fit(X, y)
        info["xgboost_cuda"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        from catboost import CatBoostClassifier

        CatBoostClassifier(task_type="GPU", iterations=5, verbose=0, allow_writing_files=False).fit(X, y)
        info["catboost_gpu"] = True
    except Exception:  # noqa: BLE001
        pass
    try:  # subprocess: LightGBM prints a C-level [Fatal] message to stderr when the GPU learner is missing
        code = ("import numpy as np, lightgbm as lgb; X=np.random.rand(300,3); y=(X[:,0]>.5).astype(int); "
                "lgb.LGBMClassifier(device='gpu', n_estimators=3, verbose=-1).fit(X, y)")
        info["lightgbm_gpu"] = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120).returncode == 0
    except Exception:  # noqa: BLE001
        pass
    return info


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.sections: Dict[str, float] = {}

    def lap(self, name: str) -> float:
        now = time.perf_counter()
        self.sections[name] = round(now - self.t0 - sum(self.sections.values()), 2)
        return self.sections[name]

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self.t0, 2)


# --------------------------------------------------------------------------------------- data + splits
def load_cohort_v2(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads the v2 cohort (builds it from the raw files if needed) and cleans SAS near-zero artefacts."""
    path = Path(path or cohort_v2_path())
    if not path.exists():
        from ml.preprocessing import add_history_features as ahf

        cohort = pd.read_csv(ahf.COHORT_PATH)
        v2, avail = ahf.build_v2_cohort(cohort)
        v2.to_csv(ahf.COHORT_V2_PATH, index=False)
        avail.to_csv(ahf.AVAILABILITY_PATH, index=False)
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number]).columns
    # XPT stores true zero in some lab columns as 5.397605e-79 (SAS special value); treat as 0.
    for c in num:
        if c not in ("SEQN",):
            df[c] = df[c].where(df[c].abs() >= 1e-30, 0.0)
    assert df["SEQN"].is_unique
    return df


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_by_cycle(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Cycle-based split: no random assignment."""
    out = {
        "train": df[df["CYCLE"].isin(TRAIN_CYCLES)],
        "validation": df[df["CYCLE"].isin(VAL_CYCLES)],
        "test": df[df["CYCLE"].isin(TEST_CYCLES)],
    }
    assert_splits_disjoint(out)
    return out


def assert_splits_disjoint(splits: Dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(splits[a]["SEQN"]) & set(splits[b]["SEQN"]):
                raise AssertionError(f"SEQN overlap between {a} and {b}")
            if set(splits[a]["CYCLE"]) & set(splits[b]["CYCLE"]):
                raise AssertionError(f"Survey cycle shared between {a} and {b}")


def split_hash(ids: Dict[str, Sequence[int]]) -> str:
    h = hashlib.sha256()
    for name in sorted(ids):
        h.update(name.encode())
        h.update(b":" + ",".join(str(int(i)) for i in sorted(ids[name])).encode() + b";")
    return h.hexdigest()


def split_summary(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, d in splits.items():
        rows.append({
            "split": name, "cycles": ", ".join(sorted(d["CYCLE"].unique())), "rows": len(d),
            "positives": int(d["target_stone"].sum()), "positive_rate_pct": round(100 * d["target_stone"].mean(), 2),
        })
    return pd.DataFrame(rows)


def freeze_splits(df: pd.DataFrame) -> Dict[str, Any]:
    """Writes the frozen row IDs (git-ignored, under ml/datasets) and a hash manifest (artifacts)."""
    splits = split_by_cycle(df)
    ids = {k: sorted(int(i) for i in v["SEQN"]) for k, v in splits.items()}
    digest = split_hash(ids)
    ensure(splits_dir())
    save_json({"hash": digest, "ids": ids}, splits_dir() / "split_ids.json")
    manifest = {
        "hash": digest,
        "cohort_file": cohort_v2_path().name,
        "cohort_sha256": file_sha256(cohort_v2_path()),
        "cycles": {"train": TRAIN_CYCLES, "validation": VAL_CYCLES, "test": TEST_CYCLES},
        "counts": {k: len(v) for k, v in ids.items()},
        "positives": {k: int(splits[k]["target_stone"].sum()) for k in splits},
        "positive_rate": {k: round(float(splits[k]["target_stone"].mean()), 4) for k in splits},
        "seed": SEED,
    }
    save_json(manifest, artifacts_dir() / "split_manifest.json")
    return manifest


def load_frozen_splits(include_test: bool = False) -> Dict[str, pd.DataFrame]:
    """Loads the frozen splits, verifying the hash against the manifest. Test rows excluded by default."""
    frozen = load_json(splits_dir() / "split_ids.json")
    manifest = load_json(artifacts_dir() / "split_manifest.json")
    if split_hash(frozen["ids"]) != frozen["hash"] or frozen["hash"] != manifest["hash"]:
        raise RuntimeError("Frozen split hash mismatch: re-run 00_data_and_splits.ipynb")
    df = load_cohort_v2()
    if file_sha256(cohort_v2_path()) != manifest["cohort_sha256"]:
        raise RuntimeError("Cohort file changed since the splits were frozen: re-run 00_data_and_splits.ipynb")
    names = ["train", "validation"] + (["test"] if include_test else [])
    out = {n: df[df["SEQN"].isin(frozen["ids"][n])].reset_index(drop=True) for n in names}
    return out


def assert_no_banned_features(features: Sequence[str]) -> None:
    bad = sorted(set(features) & set(BANNED_COLUMNS))
    if bad:
        raise AssertionError(f"Banned columns in feature list: {bad}")


# --------------------------------------------------------------------------------------- feature config
def make_variants(history_features: Sequence[str]) -> Dict[str, List[str]]:
    base = list(BASE_FEATURES)
    hist = list(history_features)
    variants = {
        "base": base,
        "base_history": base + hist,
        "base_history_sensitive": base + hist + list(SENSITIVE_FEATURES),
    }
    for feats in variants.values():
        assert_no_banned_features(feats)
        assert len(set(feats)) == len(feats)
    return variants


def load_feature_config() -> Dict[str, Any]:
    return load_json(artifacts_dir() / "feature_config.json")


def categorical_in(features: Sequence[str]) -> List[str]:
    return [f for f in features if f in CATEGORICAL_FEATURES]


def build_feature_spec(cfg: Dict[str, Any], train_df: pd.DataFrame) -> Dict[str, Any]:
    """feature_spec.json: the form definition for the registered (primary) variant."""
    primary = cfg["primary_variant"]
    feats = cfg["variants"][primary]
    out = []
    for name in feats:
        meta = dict(FEATURE_META.get(name) or HISTORY_META[name])
        is_hist = name in HISTORY_META
        col = train_df[name]
        if is_hist:
            meta["question"] = cfg["history_questions"][name]
            meta["required"] = False
            meta["default"] = None
        elif name in SENSITIVE_FEATURES:
            meta["required"] = False   # "prefer not to say" is allowed; a blank value is imputed with the training median / mode
            meta["default"] = None
        else:
            meta["required"] = meta["type"] != "derived"
            if meta["type"] in ("binary", "categorical"):
                meta["default"] = float(col.mode().iloc[0])
            else:
                meta["default"] = round(float(col.median()), 2)
        meta["name"] = name
        meta["sensitive"] = name in SENSITIVE_FEATURES
        meta["is_history"] = is_hist
        out.append(meta)
    return {
        "version": MODEL_VERSION,
        "primary_variant": primary,
        "variants": cfg["variants"],
        "features": out,
        "target": "target_stone (self-reported kidney stone, NHANES KIQ026)",
        "notes": [
            "History items are optional; an unanswered item is imputed with the training median.",
            "URDACT is derived by the server from urine albumin and urine creatinine.",
        ],
    }


# --------------------------------------------------------------------------------------- preprocessing
def build_preprocessor(features: Sequence[str]) -> ColumnTransformer:
    cats = categorical_in(features)
    nums = [f for f in features if f not in cats]
    transformers = [(
        "num",
        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
        nums,
    )]
    if cats:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cats,
        ))
    # Column order handed in must equal ``numeric + categorical`` (the bundle relies on it).
    return ColumnTransformer(transformers, remainder="drop")


def ordered_features(features: Sequence[str]) -> List[str]:
    """Numeric features first, categorical last: the exact column order of the transformed matrix."""
    cats = categorical_in(features)
    return [f for f in features if f not in cats] + cats


@dataclass
class VariantData:
    variant: str
    features: List[str]
    preprocessor: ColumnTransformer
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: List[str]


def prepare_variant(splits: Dict[str, pd.DataFrame], variant: str, features: Sequence[str]) -> VariantData:
    """Fits the preprocessor on the TRAIN rows only and transforms train + validation."""
    feats = ordered_features(features)
    assert_no_banned_features(feats)
    pre = build_preprocessor(feats).fit(splits["train"][feats])
    Xtr = np.asarray(pre.transform(splits["train"][feats]), dtype=np.float32)
    Xva = np.asarray(pre.transform(splits["validation"][feats]), dtype=np.float32)
    cats = categorical_in(feats)
    names = [f for f in feats if f not in cats]
    if cats:
        names += pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cats).tolist()
    return VariantData(variant, feats, pre, Xtr, splits["train"]["target_stone"].values.astype(int),
                       Xva, splits["validation"]["target_stone"].values.astype(int), names)


def class_weight_ratio(y: np.ndarray) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


# --------------------------------------------------------------------------------------- metrics
def _counts_by_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique thresholds (descending) with TP / FP counts for the rule ``predict positive if p >= t``."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    uniq, inv = np.unique(-p, return_inverse=True)  # ascending in -p == descending in p
    thr = -uniq
    tp = np.bincount(inv, weights=y, minlength=len(uniq)).cumsum()
    fp = np.bincount(inv, weights=1 - y, minlength=len(uniq)).cumsum()
    return thr, tp, fp


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p, dtype=float) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    denom = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {
        "threshold": float(thr), "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens, "specificity": spec, "ppv": prec, "npv": tn / max(tn + fn, 1),
        "precision": prec, "f1": f1, "mcc": mcc, "positive_rate": float(pred.mean()) if len(pred) else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def threshold_max_balanced_accuracy(y_val: np.ndarray, p_val: np.ndarray) -> Dict[str, float]:
    """Rule 1 (the registered operating threshold): maximise balanced accuracy on VALIDATION data."""
    thr, tp, fp = _counts_by_threshold(y_val, p_val)
    P, N = float(np.sum(y_val)), float(len(y_val) - np.sum(y_val))
    bal = (tp / P + (N - fp) / N) / 2
    k = int(np.argmax(bal))  # first max == highest threshold among ties
    return metrics_at_threshold(y_val, p_val, float(thr[k]))


def threshold_for_accuracy(y_val: np.ndarray, p_val: np.ndarray, min_accuracy: float = ACC_TARGET_VALIDATION) -> Optional[Dict[str, float]]:
    """Rule 2: among thresholds whose VALIDATION accuracy >= min_accuracy, take the one with highest
    sensitivity (the lowest such threshold). Returns None when no threshold reaches the target."""
    thr, tp, fp = _counts_by_threshold(y_val, p_val)
    n = len(y_val)
    P = float(np.sum(y_val))
    acc = (tp + ((n - P) - fp)) / n
    ok = np.where(acc >= min_accuracy)[0]
    if len(ok) == 0:
        return None
    k = int(ok[-1])  # thresholds descend, so the last qualifying index is the lowest threshold
    return metrics_at_threshold(y_val, p_val, float(thr[k]))


def operating_points(y_val: np.ndarray, p_val: np.ndarray) -> Dict[str, Optional[Dict[str, float]]]:
    """Both operating points, derived from VALIDATION scores only (the caller must never pass test scores)."""
    return {OP_BALANCED: threshold_max_balanced_accuracy(y_val, p_val),
            OP_ACCURACY: threshold_for_accuracy(y_val, p_val, ACC_TARGET_VALIDATION)}


def _rank_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


rank_metrics = _rank_metrics


def evaluate_scores(y: np.ndarray, p: np.ndarray, thresholds: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Point estimates: ranking / calibration metrics plus accuracy-type metrics at each threshold."""
    y = np.asarray(y).astype(int)
    prev = float(y.mean())
    out: Dict[str, Any] = {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": prev,
        "majority_baseline_accuracy": 1.0 - prev, **_rank_metrics(y, p), "at_threshold": {},
    }
    for name, t in thresholds.items():
        out["at_threshold"][name] = None if t is None else metrics_at_threshold(y, p, t)
    return out


def bootstrap_metrics(
    y: np.ndarray, p: np.ndarray, thresholds: Dict[str, Optional[float]], n_boot: int = 1000, seed: int = SEED
) -> Dict[str, Any]:
    """Point estimate + percentile 95% CI over ``n_boot`` row-level bootstrap resamples."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(y)
    keys = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "positive_rate"]
    rank = {k: [] for k in ("roc_auc", "pr_auc", "brier")}
    thr_stats = {t: {k: [] for k in keys} for t, v in thresholds.items() if v is not None}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        for k, v in _rank_metrics(yb, pb).items():
            rank[k].append(v)
        for name in thr_stats:
            m = metrics_at_threshold(yb, pb, thresholds[name])
            for k in keys:
                thr_stats[name][k].append(m[k])

    def ci(arr: List[float]) -> List[float]:
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    point = evaluate_scores(y, p, thresholds)
    result: Dict[str, Any] = {k: v for k, v in point.items() if k != "at_threshold"}
    result["n_bootstrap"] = n_boot
    for k in rank:
        result[f"{k}_ci95"] = ci(rank[k])
    result["at_threshold"] = {}
    for name, m in point["at_threshold"].items():
        if m is None:
            result["at_threshold"][name] = None
            continue
        entry = dict(m)
        for k in keys:
            entry[f"{k}_ci95"] = ci(thr_stats[name][k])
        result["at_threshold"][name] = entry
    return result


# --------------------------------------------------------------------------------------- calibration
def choose_calibrator(p_raw_val: np.ndarray, y_val: np.ndarray, seed: int = SEED, n_splits: int = 5) -> Tuple[ProbCalibrator, Dict[str, Any]]:
    """Isotonic vs sigmoid, picked by Brier score on the validation cycle.

    The comparison uses cross-validated Brier inside the validation set (isotonic would always win
    in-sample); the winner is then refit on all validation rows and frozen."""
    y_val = np.asarray(y_val).astype(int)
    p_raw_val = np.asarray(p_raw_val, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_brier = {}
    for method in ("isotonic", "sigmoid"):
        oof = np.zeros(len(y_val))
        for tr, te in skf.split(p_raw_val, y_val):
            oof[te] = ProbCalibrator.fit(p_raw_val[tr], y_val[tr], method).predict(p_raw_val[te])
        cv_brier[method] = float(brier_score_loss(y_val, oof))
    method = min(cv_brier, key=cv_brier.get)
    cal = ProbCalibrator.fit(p_raw_val, y_val, method)
    report = {
        "method": method,
        "cv_brier_isotonic": cv_brier["isotonic"],
        "cv_brier_sigmoid": cv_brier["sigmoid"],
        "brier_raw_uncalibrated": float(brier_score_loss(y_val, np.clip(p_raw_val, 0, 1))),
        "brier_calibrated_in_sample": float(brier_score_loss(y_val, cal.predict(p_raw_val))),
    }
    return cal, report


# --------------------------------------------------------------------------------------- tuning / CV
def random_search(
    sample_params: Callable[[np.random.RandomState], Dict[str, Any]],
    fit_eval: Callable[[Dict[str, Any]], Dict[str, Any]],
    n_trials: int,
    seed: int = SEED,
    first_params: Optional[Dict[str, Any]] = None,
    score_key: str = "val_roc_auc",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Random search maximising ``fit_eval(params)[score_key]`` (validation cycle, never the test cycle)."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_trials):
        params = first_params if (i == 0 and first_params) else sample_params(rng)
        res = fit_eval(params)
        rows.append({"trial": i, "params": json.dumps(jsonable(params)), **{k: v for k, v in res.items()}})
    df = pd.DataFrame(rows)
    best = df.sort_values(score_key, ascending=False, kind="stable").iloc[0]
    return json.loads(best["params"]), df


def loco_cv(
    dev_df: pd.DataFrame,
    features: Sequence[str],
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    cycles: Sequence[str],
    flag_rates: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Leave-one-cycle-out CV over ``cycles``: train on the other cycles, score the held-out cycle.
    The preprocessor is refit on each fold's training cycles. Scores are uncalibrated ranking metrics.

    ``flag_rates`` ({operating point: fraction of people flagged}) adds per-fold accuracy / sensitivity: in every
    fold the top ``rate`` fraction of held-out scores is flagged. The rates come from the validation-chosen
    operating points, so no label of the held-out fold is used to place the cut-off (fold models are
    uncalibrated and differ from the frozen model, so a probability cut-off would not transfer)."""
    feats = ordered_features(features)
    folds = []
    for held in cycles:
        tr = dev_df[dev_df["CYCLE"].isin([c for c in cycles if c != held])]
        ho = dev_df[dev_df["CYCLE"] == held]
        pre = build_preprocessor(feats).fit(tr[feats])
        Xtr = np.asarray(pre.transform(tr[feats]), dtype=np.float32)
        Xho = np.asarray(pre.transform(ho[feats]), dtype=np.float32)
        p = fit_predict(Xtr, tr["target_stone"].values.astype(int), Xho)
        yho = ho["target_stone"].values.astype(int)
        fold = {"held_out_cycle": held, "n": len(ho), "prevalence": float(yho.mean()), **_rank_metrics(yho, p)}
        for pt, rate in (flag_rates or {}).items():
            thr = float(np.quantile(p, 1.0 - rate))
            m = metrics_at_threshold(yho, p, thr)
            fold[f"acc_{pt}"], fold[f"sens_{pt}"], fold[f"spec_{pt}"] = m["accuracy"], m["sensitivity"], m["specificity"]
            fold[f"flagged_{pt}"] = m["positive_rate"]
        folds.append(fold)
    out = {"folds": folds, **spread([f["roc_auc"] for f in folds], "roc_auc"), **spread([f["pr_auc"] for f in folds], "pr_auc")}
    for pt in (flag_rates or {}):
        out.update(spread([f[f"acc_{pt}"] for f in folds], f"acc_{pt}"))
        out.update(spread([f[f"sens_{pt}"] for f in folds], f"sens_{pt}"))
    out["flag_rates"] = dict(flag_rates or {})
    return out


def grouped_cv(
    dev_df: pd.DataFrame,
    features: Sequence[str],
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Optional robustness check: StratifiedGroupKFold with group = cycle + SDMVSTRA + SDMVPSU."""
    feats = ordered_features(features)
    groups = (dev_df["CYCLE"].astype(str) + "_" + dev_df["SDMVSTRA"].astype(int).astype(str) + "_"
              + dev_df["SDMVPSU"].astype(int).astype(str)).values
    y = dev_df["target_stone"].values.astype(int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, aps = [], []
    for tr_idx, te_idx in sgkf.split(dev_df, y, groups):
        tr, te = dev_df.iloc[tr_idx], dev_df.iloc[te_idx]
        assert not (set(groups[tr_idx]) & set(groups[te_idx]))
        pre = build_preprocessor(feats).fit(tr[feats])
        Xtr = np.asarray(pre.transform(tr[feats]), dtype=np.float32)
        Xte = np.asarray(pre.transform(te[feats]), dtype=np.float32)
        p = fit_predict(Xtr, y[tr_idx], Xte)
        aucs.append(roc_auc_score(y[te_idx], p))
        aps.append(average_precision_score(y[te_idx], p))
    return {**spread(aucs, "roc_auc"), **spread(aps, "pr_auc"), "n_splits": n_splits}


def spread(values: Sequence[float], name: str) -> Dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return {f"{name}_mean": float(v.mean()), f"{name}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            f"{name}_min": float(v.min()), f"{name}_max": float(v.max())}


# --------------------------------------------------------------------------------------- validation halves
def split_validation(seqn: Sequence[int], y_val: np.ndarray, es_fraction: float = 0.5, seed: int = SEED) -> Dict[str, Any]:
    """Splits the VALIDATION cycle (by SEQN, stratified on the label) into two disjoint halves.

    ``es``  : early stopping and hyper-parameter scoring.
    ``cal`` : everything that is selected or reported on validation: imbalance strategy, calibration method /
              calibrator, both operating thresholds and all validation metric tables.
    Keeping the halves apart means the rows that decide *where to stop boosting* are never the rows used to
    calibrate probabilities or place a threshold. Neither half is ever resampled."""
    from sklearn.model_selection import train_test_split

    seqn = np.asarray(seqn)
    idx = np.arange(len(seqn))
    es_idx, cal_idx = train_test_split(idx, train_size=es_fraction, stratify=np.asarray(y_val).astype(int), random_state=seed)
    es_idx, cal_idx = np.sort(es_idx), np.sort(cal_idx)
    assert not set(seqn[es_idx]) & set(seqn[cal_idx])
    digest = hashlib.sha256((",".join(map(str, sorted(int(i) for i in seqn[es_idx]))) + "|"
                             + ",".join(map(str, sorted(int(i) for i in seqn[cal_idx])))).encode()).hexdigest()
    return {"es": es_idx, "cal": cal_idx, "hash": digest}


# --------------------------------------------------------------------------------------- class imbalance
def categorical_column_indices(feature_names: Sequence[str]) -> List[int]:
    """Indices (in the transformed matrix) of columns that must stay valid categories under SMOTENC:
    sex, the yes/no (and diabetes 0 / 0.5 / 1) history items and the one-hot race columns."""
    cats = []
    for i, name in enumerate(feature_names):
        if name == "RIAGENDR" or name in HISTORY_META or name.startswith("RIDRETH1_"):
            cats.append(i)
    return cats


def make_sampler(strategy: str, cat_idx: Sequence[int], seed: int = SEED) -> Any:
    from imblearn.over_sampling import SMOTENC, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    if strategy in ("none", "class_weight"):
        return "passthrough"
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=seed)
    if strategy == "smotenc":
        return SMOTENC(categorical_features=list(cat_idx), random_state=seed, k_neighbors=5)
    raise ValueError(f"Unknown imbalance strategy: {strategy}")


def fit_with_strategy(estimator: Any, strategy: str, X: np.ndarray, y: np.ndarray, cat_idx: Sequence[int],
                      fit_params: Optional[Dict[str, Any]] = None, seed: int = SEED) -> Any:
    """Fits ``imblearn.Pipeline([sampler, clf])`` on TRAINING rows. The sampler only acts inside ``fit``;
    ``predict_proba`` and any eval_set passed in ``fit_params`` (validation rows) are never resampled.
    Class weighting is the estimator's own job (``class_weight`` strategy), not a sampler."""
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipe = ImbPipeline([("sampler", make_sampler(strategy, cat_idx, seed)), ("clf", estimator)])
    pipe.fit(X, y, **{f"clf__{k}": v for k, v in (fit_params or {}).items()})
    return pipe


def calibrated_oof_metrics(p_raw: np.ndarray, y: np.ndarray, seed: int = SEED, n_splits: int = 5) -> Dict[str, Any]:
    """Calibrates with the best of isotonic / sigmoid (by CV Brier) and scores cross-validated calibrated
    probabilities, so a strategy is judged on how well *calibrated* probabilities rank and fit, out of sample."""
    y = np.asarray(y).astype(int)
    p_raw = np.asarray(p_raw, dtype=float)
    _, rep = choose_calibrator(p_raw, y, seed=seed, n_splits=n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(p_raw, y):
        oof[te] = ProbCalibrator.fit(p_raw[tr], y[tr], rep["method"]).predict(p_raw[te])
    return {"calibration_method": rep["method"], "oof_roc_auc": float(roc_auc_score(y, oof)),
            "oof_pr_auc": float(average_precision_score(y, oof)), "oof_brier": float(brier_score_loss(y, oof)),
            "mean_raw_risk": float(np.clip(p_raw, 0, 1).mean()), "mean_calibrated_risk": float(oof.mean()), "prevalence": float(y.mean())}


def select_strategy(table: pd.DataFrame, tie_band: float = STRATEGY_TIE_BAND) -> Tuple[str, Dict[str, Any]]:
    """Pre-declared rule (validation only): best post-calibration PR-AUC; strategies within ``tie_band`` of it are
    tied and the lowest post-calibration Brier wins; remaining ties go to the simplest strategy (list order)."""
    best = float(table["oof_pr_auc"].max())
    tied = table[table["oof_pr_auc"] >= best - tie_band]
    order = {s_: i for i, s_ in enumerate(IMBALANCE_STRATEGIES)}
    pick = sorted(tied.index, key=lambda k: (round(float(tied.loc[k, "oof_brier"]), 5), order[k]))[0]
    info = {"selected": pick, "best_pr_auc": best, "tied_with": [k for k in tied.index if k != pick],
            "none_within_tie_band": bool("none" in tied.index), "tie_band": tie_band,
            "pr_auc_gain_over_none": float(table.loc[pick, "oof_pr_auc"] - table.loc["none", "oof_pr_auc"])}
    return pick, info


# --------------------------------------------------------------------------------------- metric tables
METRIC_COLUMNS = ["ROC-AUC", "PR-AUC", "Brier", "Accuracy", "Balanced acc", "Sensitivity", "Specificity", "Precision",
                  "F1", "MCC", "Majority acc", "Pos-pred rate", "TN", "FP", "FN", "TP"]
_CI_KEYS = {"ROC-AUC": "roc_auc", "PR-AUC": "pr_auc", "Brier": "brier", "Accuracy": "accuracy", "Balanced acc": "balanced_accuracy",
            "Sensitivity": "sensitivity", "Specificity": "specificity", "Precision": "precision", "F1": "f1", "MCC": "mcc",
            "Pos-pred rate": "positive_rate"}


def metric_row(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    """One table row: ranking / calibration metrics of the score vector plus threshold metrics at ``thr``."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    m = metrics_at_threshold(y, p, thr)
    return {"ROC-AUC": float(roc_auc_score(y, p)), "PR-AUC": float(average_precision_score(y, p)),
            "Brier": float(brier_score_loss(y, np.clip(p, 0, 1))), "Accuracy": m["accuracy"], "Balanced acc": m["balanced_accuracy"],
            "Sensitivity": m["sensitivity"], "Specificity": m["specificity"], "Precision": m["precision"], "F1": m["f1"],
            "MCC": m["mcc"], "Majority acc": 1.0 - float(y.mean()), "Pos-pred rate": m["positive_rate"],
            "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"]}


def _fmt(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}"


def show_metrics_table(
    entries: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]], title: str, name: Optional[str] = None,
    thresholds: Optional[Dict[str, Dict[str, Optional[float]]]] = None, points: Sequence[str] = OPERATING_POINTS,
    ci: bool = False, results: Optional[Dict[str, Dict[str, Any]]] = None, extra: Optional[Dict[str, Dict[str, Any]]] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """THE metric table used by every model notebook: prints it (one block, header ``METRICS TABLE: <title>``) and saves it to
    ``ml/outputs/metrics/nhanes_v003/tables/<name>.csv``.

    entries    : {label: (y, p)} score vectors. Without ``thresholds`` the two operating thresholds are derived from the
                 given scores themselves, which is only legitimate for VALIDATION scores; for any other data pass the
                 frozen ``thresholds`` ({label: {point: thr}}) or ``results``.
    ci         : adds bootstrap 95% CIs (1,000 resamples) - requires frozen ``thresholds``.
    results    : {label: bootstrap_metrics() output} already computed (e.g. by ``final_test_evaluation``); the table is
                 built from them without touching any data.
    Columns: ROC-AUC, PR-AUC, Brier, Accuracy, Balanced acc, Sensitivity, Specificity, Precision, F1, MCC, Majority acc
    (accuracy of 'no stone for everyone'), Pos-pred rate and the confusion-matrix counts."""
    rows = []
    if results is not None:
        for label, res in results.items():
            for pt in points:
                at = res["at_threshold"].get(pt)
                if at is None:
                    rows.append({"model/strategy": label, "point": pt, "Threshold": "unreachable"})
                    continue
                row = {"model/strategy": label, "point": pt, "Threshold": _fmt(at["threshold"], 4)}
                for col in METRIC_COLUMNS:
                    key = _CI_KEYS.get(col)
                    if key is None:
                        val = {"Majority acc": res["majority_baseline_accuracy"], "TN": at["tn"], "FP": at["fp"], "FN": at["fn"], "TP": at["tp"]}[col]
                        row[col] = _fmt(val) if isinstance(val, float) else int(val)
                        continue
                    src = at if key in at else res
                    lo_hi = src.get(f"{key}_ci95")
                    d = 4 if col == "Brier" else 3
                    row[col] = f"{_fmt(src[key], d)} [{_fmt(lo_hi[0], d)}, {_fmt(lo_hi[1], d)}]" if lo_hi else _fmt(src[key], d)
                rows.append(row)
    else:
        for label, (y, p) in (entries or {}).items():
            y = np.asarray(y).astype(int)
            p = np.asarray(p, dtype=float)
            if thresholds is not None:
                thr = thresholds[label]
            else:
                if ci:
                    raise ValueError("CIs need frozen thresholds (never derive thresholds on the data being evaluated).")
                ops = operating_points(y, p)
                thr = {k: (None if v is None else v["threshold"]) for k, v in ops.items()}
            boot = bootstrap_metrics(y, p, {k: thr.get(k) for k in points}, n_boot=1000) if ci else None
            for pt in points:
                if thr.get(pt) is None:
                    rows.append({"model/strategy": label, "point": pt, "Threshold": "unreachable"})
                    continue
                r = metric_row(y, p, thr[pt])
                row = {"model/strategy": label, "point": pt, "Threshold": _fmt(thr[pt], 4)}
                for col in METRIC_COLUMNS:
                    if boot is not None and col in _CI_KEYS:
                        key = _CI_KEYS[col]
                        src = boot["at_threshold"][pt] if key in boot["at_threshold"][pt] else boot
                        lo, hi = src[f"{key}_ci95"]
                        d = 4 if col == "Brier" else 3
                        row[col] = f"{_fmt(r[col], d)} [{_fmt(lo, d)}, {_fmt(hi, d)}]"
                    else:
                        row[col] = r[col] if col in ("TN", "FP", "FN", "TP") else _fmt(r[col], 4 if col == "Brier" else 3)
                rows.append(row)
    for row in rows:
        for k, v in (extra or {}).get(row["model/strategy"], {}).items():
            row[k] = v
    df = pd.DataFrame(rows)
    if name:
        out = ensure(metrics_dir() / "tables") / f"{name}.csv"
        df.to_csv(out, index=False)
    if not quiet:
        print(f"=== METRICS TABLE: {title} ===")
        with pd.option_context("display.width", 400, "display.max_columns", 60, "display.max_colwidth", 40):
            print(df.to_string(index=False))
    return df


def plot_confusion_matrices(y: np.ndarray, p: np.ndarray, thresholds: Dict[str, Optional[float]], path: Path, title: str) -> Path:
    """One confusion-matrix panel per operating point (counts and row-normalised percentages)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [k for k in OPERATING_POINTS if thresholds.get(k) is not None]
    fig, axes = plt.subplots(1, len(pts), figsize=(4.6 * len(pts), 4.2), squeeze=False)
    for ax, pt in zip(axes[0], pts):
        m = metrics_at_threshold(y, p, thresholds[pt])
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({100 * norm[i, j]:.0f}% of row)", ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else "black", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred no stone", "pred stone"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["no stone", "stone"])
        ax.set_title(f"{pt} point (thr {thresholds[pt]:.3f})\nacc {m['accuracy']:.3f} | sens {m['sensitivity']:.3f} | spec {m['specificity']:.3f}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    ensure(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------- provenance
def library_versions() -> Dict[str, str]:
    import importlib

    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "sklearn", "imblearn", "xgboost", "catboost", "lightgbm", "interpret", "shap", "joblib"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "not installed"
    return out


def provenance(notebook_file: str, timer: "Timer", device: Dict[str, Any], split_hash_value: str, training_seconds: float) -> Dict[str, Any]:
    """Recorded in every metrics JSON; the notebook tests compare it with the executed .ipynb."""
    return {
        "notebook": notebook_file,
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device, "training_seconds": training_seconds, "notebook_seconds": timer.total,
        "libraries": library_versions(), "split_hash": split_hash_value, "seed": SEED,
    }


# --------------------------------------------------------------------------------------- leakage audit
def leakage_audit(df: pd.DataFrame, features: Sequence[str], flag_above: float = 0.85) -> pd.DataFrame:
    """Single-feature ROC-AUC (direction-agnostic) on the given rows. Flags any feature above 0.85."""
    y = df["target_stone"].values.astype(int)
    rows = []
    for f in features:
        v = df[f].astype(float)
        v = v.fillna(v.median()).values
        auc = roc_auc_score(y, v)
        auc = max(auc, 1 - auc)
        rows.append({"feature": f, "single_feature_auc": round(float(auc), 4), "flagged": bool(auc > flag_above)})
    return pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------- bundles + test-once
def make_bundle(
    model_name: str, family: str, variant: str, vd: VariantData, estimator: Any,
    calibrator: ProbCalibrator, thresholds: Dict[str, Optional[float]], metadata: Optional[Dict[str, Any]] = None,
) -> NhanesBundle:
    return NhanesBundle(
        model_name=model_name, family=family, variant=variant, input_features=vd.features,
        categorical_features=categorical_in(vd.features), preprocessor=vd.preprocessor,
        estimator=estimator, calibrator=calibrator, thresholds=thresholds, metadata=metadata,
    )


def frozen_fingerprint(model_name: str, variant: str, params: Dict[str, Any], features: Sequence[str],
                       calibration_method: str, thresholds: Dict[str, Optional[float]], split_hash_value: str) -> str:
    payload = json.dumps(jsonable({
        "model": model_name, "variant": variant, "params": params, "features": list(features),
        "calibration": calibration_method, "thresholds": {k: (None if v is None else round(v, 4)) for k, v in thresholds.items()},
        "split": split_hash_value,
    }), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def final_test_evaluation(bundle: NhanesBundle, model_name: str, variant: str, fingerprint: str) -> Dict[str, Any]:
    """The ONLY entry point to the test cycle. Evaluates the frozen bundle once, with 1,000-sample bootstrap CIs.

    A ledger (ml/outputs/metrics/nhanes_v003/test_ledger.json) records the result. Re-running returns the
    stored result for an identical frozen model; a changed model is refused unless NHANES_ALLOW_TEST_REEVAL=1.
    In dry-run mode the VALIDATION cycle is scored instead and nothing is recorded."""
    thresholds = {k: v for k, v in bundle.thresholds.items()}
    key = f"{model_name}::{variant}"
    if is_dry_run():
        d = load_frozen_splits(include_test=False)["validation"]
        p = bundle.predict_proba(d)
        res = bootstrap_metrics(d["target_stone"].values, p, thresholds, n_boot=100)
        res.update({"dry_run": True, "evaluated_on": "validation (dry run)", "fingerprint": fingerprint, "cached": False})
        ensure(predictions_dir())
        np.savez_compressed(predictions_dir() / f"{model_name}__{variant}__test.npz",
                            y=d["target_stone"].values, p=p, seqn=d["SEQN"].values)
        return {"result": res, "y": d["target_stone"].values, "p": p, "seqn": d["SEQN"].values}
    ledger_path = metrics_dir() / "test_ledger.json"
    ledger = load_json(ledger_path) if ledger_path.exists() else {}
    pred_path = predictions_dir() / f"{model_name}__{variant}__test.npz"
    if key in ledger:
        if ledger[key]["fingerprint"] == fingerprint and pred_path.exists():
            print(f"[test ledger] {key}: frozen model unchanged - returning the stored result, test set NOT re-evaluated.")
            z = np.load(pred_path)
            res = dict(ledger[key]["result"], cached=True)
            return {"result": res, "y": z["y"], "p": z["p"], "seqn": z["seqn"]}
        if os.environ.get("NHANES_ALLOW_TEST_REEVAL", "0") != "1":
            raise RuntimeError(
                f"{key} was already evaluated on the test cycle with a different frozen model. Re-evaluating would "
                "mean tuning on the test set. Set NHANES_ALLOW_TEST_REEVAL=1 only for a documented, intentional re-run."
            )
    test = load_frozen_splits(include_test=True)["test"]
    p = bundle.predict_proba(test)
    y = test["target_stone"].values.astype(int)
    res = bootstrap_metrics(y, p, thresholds, n_boot=1000)
    res.update({"dry_run": False, "evaluated_on": "test (" + ", ".join(TEST_CYCLES) + ")", "fingerprint": fingerprint, "cached": False})
    ensure(predictions_dir())
    np.savez_compressed(pred_path, y=y, p=p, seqn=test["SEQN"].values)
    ledger[key] = {"fingerprint": fingerprint, "result": res, "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_json(ledger, ledger_path)
    return {"result": res, "y": y, "p": p, "seqn": test["SEQN"].values}


# --------------------------------------------------------------------------------------- plots
def plot_calibration(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path, title: str, n_bins: int = 10) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for label, (y, p) in curves.items():
        frac, mean_p = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_p, frac, "o-", ms=4, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed stone fraction")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_importance(names: Sequence[str], values: Sequence[float], path: Path, title: str, top: int = 15) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh([names[i] for i in order], [values[i] for i in order], color="#1f7a8c")
    ax.set_xlabel("Mean |contribution| (log-odds, validation cycle)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    ensure(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def global_importance(bundle: NhanesBundle, df: pd.DataFrame, max_rows: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """Mean |local contribution| per raw feature on (a sample of) the given rows."""
    sample = df.sample(n=min(max_rows, len(df)), random_state=seed)
    contrib = bundle.contributions(sample)
    return (pd.DataFrame({"feature": bundle.input_features, "mean_abs_contribution": np.abs(contrib).mean(axis=0)})
            .sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------------------- registry
OP_RULE = {
    OP_BALANCED: "maximise balanced accuracy on the validation cycle (2015-2016)",
    OP_ACCURACY: f"maximise sensitivity subject to validation accuracy >= {int(ACC_TARGET_VALIDATION * 100)}% (validation cycle 2015-2016)",
}
OP_LABEL = {OP_ACCURACY: "Accuracy-focused (>70% accuracy)", OP_BALANCED: "Balanced (max balanced accuracy)"}


def registry_entry_fields(model_result: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """Manifest fields for one model's primary variant (read from its saved metrics JSON): both operating points
    with their thresholds, validation metrics and held-out test metrics (with bootstrap CIs)."""
    v = model_result["variants"][variant]
    test = v["test"]
    at = test["at_threshold"]
    keep = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "positive_rate")

    def pick(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if entry is None:
            return None
        out = {"threshold": entry["threshold"], "tp": entry["tp"], "fp": entry["fp"], "tn": entry["tn"], "fn": entry["fn"]}
        for k in keep:
            out[k] = entry[k]
            if f"{k}_ci95" in entry:
                out[f"{k}_ci95"] = entry[f"{k}_ci95"]
        return out

    def val_pick(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return None if entry is None else {k: entry[k] for k in ("threshold", *keep, "tp", "fp", "tn", "fn")}

    thr = v["thresholds"]
    return {
        "status": "pending_review",
        "is_deployed": False,
        "dataset_name": "NHANES Kidney Stone Cohort v2 (2007-2018) with self-reported history",
        "artifact_format": "nhanes_bundle_v3",
        "variant": variant,
        "default_operating_point": OP_ACCURACY,
        "threshold_rule": OP_RULE[OP_ACCURACY],
        "thresholds": {pt: (None if thr[pt] is None else thr[pt]["threshold"]) for pt in OPERATING_POINTS},
        "operating_points": {pt: {"label": OP_LABEL[pt], "rule": OP_RULE[pt],
                                  "threshold": None if thr[pt] is None else thr[pt]["threshold"],
                                  "validation": val_pick(thr[pt]), "test": pick(at[pt])} for pt in OPERATING_POINTS},
        "imbalance_strategy": v["imbalance"]["selected"],
        "split_counts": model_result["split_counts"],
        "split_hash": model_result["split_hash"],
        "device": model_result["device"],
        "calibration_method": v["calibration"]["method"],
        "provenance": model_result["provenance"],
        "test_metrics": {
            "evaluated_on": test["evaluated_on"],
            "test_cycle": ", ".join(TEST_CYCLES),
            "n": test["n"], "positives": test["positives"], "prevalence": test["prevalence"],
            "majority_baseline_accuracy": test["majority_baseline_accuracy"],
            "roc_auc": test["roc_auc"], "roc_auc_ci95": test["roc_auc_ci95"],
            "pr_auc": test["pr_auc"], "pr_auc_ci95": test["pr_auc_ci95"],
            "brier": test["brier"], "brier_ci95": test["brier_ci95"],
            "at_balanced_threshold": pick(at[OP_BALANCED]),
            "at_accuracy_threshold": pick(at[OP_ACCURACY]),
        },
        "trained_at": model_result["trained_at"],
    }
