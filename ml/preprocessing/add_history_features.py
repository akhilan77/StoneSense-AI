"""Build the v2 NHANES cohort: add self-reported history features, drop KIQ029.

Usage (from repo root):
    python ml/preprocessing/add_history_features.py

Reads BPQ_*, DIQ_*, MCQ_* XPT files for cycles E-J from ml/datasets/nhanes_raw/, keeps
BPQ020, BPQ080, DIQ010, MCQ080, MCQ520, MCQ550, decodes NHANES special codes
(7 = refused, 9 = don't know -> missing), recodes yes/no to 1/0 (DIQ010: 3 = borderline -> 0.5),
and left-merges them (one-to-one on SEQN) into ml/datasets/nhanes_kidney_stone_cohort.csv.

Writes (never overwriting the original cohort):
    ml/datasets/nhanes_kidney_stone_cohort_v2.csv
    ml/datasets/history_feature_availability.csv   (cycle, variable, present, percent_missing, label)

KIQ029 (times a stone was passed in the past year) is dropped: it is only collected in 2015-2018
and is a post-outcome variable of the label.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "ml" / "datasets"
RAW_DIR = DATASETS_DIR / "nhanes_raw"
COHORT_PATH = DATASETS_DIR / "nhanes_kidney_stone_cohort.csv"
COHORT_V2_PATH = DATASETS_DIR / "nhanes_kidney_stone_cohort_v2.csv"
AVAILABILITY_PATH = DATASETS_DIR / "history_feature_availability.csv"

CYCLE_SUFFIX = {
    "2007-2008": "E",
    "2009-2010": "F",
    "2011-2012": "G",
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}

# variable -> (source file prefix, plain-language question used by the frontend form)
HISTORY_VARS: Dict[str, Tuple[str, str]] = {
    "BPQ020": ("BPQ", "Has a doctor or other health professional ever told you that you have high blood pressure?"),
    "BPQ080": ("BPQ", "Has a doctor or other health professional ever told you that your blood cholesterol level was high?"),
    "DIQ010": ("DIQ", "Has a doctor or other health professional ever told you that you have diabetes or sugar diabetes?"),
    "MCQ080": ("MCQ", "Has a doctor or other health professional ever told you that you were overweight?"),
    "MCQ520": ("MCQ", "During the past 12 months, have you had abdominal pain?"),
    "MCQ550": ("MCQ", "Has a doctor or other health professional ever told you that you have gallstones?"),
}
HISTORY_COLUMNS: List[str] = list(HISTORY_VARS)

# Value recoding after special codes (7, 9) became NaN.
YES_NO = {1.0: 1.0, 2.0: 0.0}
DIQ010_MAP = {1.0: 1.0, 2.0: 0.0, 3.0: 0.5}  # 3 = borderline


def decode_history_series(name: str, s: pd.Series) -> pd.Series:
    """7/9 (refused / don't know) -> NaN, then recode to 1/0 (DIQ010 also 0.5 for borderline)."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s.where(~s.isin([7.0, 9.0]))
    mapping = DIQ010_MAP if name == "DIQ010" else YES_NO
    return s.map(mapping)


def _read_xpt(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Reads an XPT file and returns (frame, {column: label}); raises if it is not a valid XPT."""
    reader = pd.read_sas(path, format="xport", iterator=True)
    labels = {f["name"].decode().strip(): f["label"].decode().strip() for f in reader.fields}
    df = reader.read()
    reader.close()
    if "SEQN" not in df.columns:
        raise ValueError(f"{path.name} has no SEQN column")
    return df, labels


def load_history_for_cycle(raw_dir: Path, suffix: str) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Returns (SEQN + decoded history columns, {variable: xpt label or None if absent})."""
    frames = []
    labels: Dict[str, Optional[str]] = {v: None for v in HISTORY_COLUMNS}
    for prefix in sorted({p for p, _ in HISTORY_VARS.values()}):
        df, lab = _read_xpt(raw_dir / f"{prefix}_{suffix}.xpt")
        if not df["SEQN"].is_unique:
            raise AssertionError(f"{prefix}_{suffix}: SEQN is not unique")
        keep = ["SEQN"] + [v for v, (p, _) in HISTORY_VARS.items() if p == prefix and v in df.columns]
        sub = df[keep].copy()
        sub["SEQN"] = sub["SEQN"].astype(int)
        for v in keep[1:]:
            sub[v] = decode_history_series(v, sub[v])
            labels[v] = lab.get(v)
        frames.append(sub.set_index("SEQN"))
    merged = pd.concat(frames, axis=1).reset_index()
    for v in HISTORY_COLUMNS:  # variable not collected in this cycle -> all-missing column
        if v not in merged.columns:
            merged[v] = np.nan
    return merged[["SEQN"] + HISTORY_COLUMNS], labels


def build_v2_cohort(cohort: pd.DataFrame, raw_dir: Path = RAW_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (v2 cohort, availability table). Asserts row and label counts are unchanged."""
    n_rows, n_pos = len(cohort), int(cohort["target_stone"].sum())
    pos_by_cycle = cohort.groupby("CYCLE")["target_stone"].sum().to_dict()

    hist_parts, label_rows = [], []
    for cycle, suffix in CYCLE_SUFFIX.items():
        hist, labels = load_history_for_cycle(raw_dir, suffix)
        hist["CYCLE"] = cycle
        hist_parts.append(hist)
        for v in HISTORY_COLUMNS:
            label_rows.append({"cycle": cycle, "variable": v, "label": labels[v] or "", "present": labels[v] is not None})
    hist_all = pd.concat(hist_parts, ignore_index=True)
    # SEQN is unique across cycles in NHANES; verify before a one-to-one merge.
    if not hist_all["SEQN"].is_unique:
        raise AssertionError("History SEQN is not unique across cycles")

    base = cohort.drop(columns=[c for c in ["KIQ029"] + HISTORY_COLUMNS if c in cohort.columns])
    v2 = base.merge(hist_all.drop(columns=["CYCLE"]), on="SEQN", how="left", validate="one_to_one")

    assert len(v2) == n_rows, "Row count changed after merge"
    assert int(v2["target_stone"].sum()) == n_pos, "Positive count changed after merge"
    assert v2.groupby("CYCLE")["target_stone"].sum().to_dict() == pos_by_cycle, "Per-cycle positives changed"
    assert "KIQ029" not in v2.columns

    rows = []
    info = pd.DataFrame(label_rows).set_index(["cycle", "variable"])
    for cycle in CYCLE_SUFFIX:
        sub = v2[v2["CYCLE"] == cycle]
        for v in HISTORY_COLUMNS:
            rows.append({
                "cycle": cycle,
                "variable": v,
                "present": bool(info.loc[(cycle, v), "present"]),
                "percent_missing": round(float(sub[v].isna().mean() * 100), 2),
                "label": info.loc[(cycle, v), "label"],
            })
    return v2, pd.DataFrame(rows)


def select_history_features(
    availability: pd.DataFrame,
    train_cycles: List[str],
    val_cycles: List[str],
    test_cycles: List[str],
    max_missing_pct: float = 95.0,
) -> Dict[str, Dict[str, object]]:
    """Per-variable decision. A variable is usable only if it is present, with the same XPT label,
    in every train, validation and test cycle (and not entirely missing in any of them).
    No whole cycle is ever imputed."""
    needed = list(train_cycles) + list(val_cycles) + list(test_cycles)
    decisions: Dict[str, Dict[str, object]] = {}
    for v in HISTORY_COLUMNS:
        sub = availability[(availability["variable"] == v) & (availability["cycle"].isin(needed))]
        absent = sorted(sub.loc[~sub["present"], "cycle"])
        labels = {str(x).lower() for x in sub.loc[sub["present"], "label"]}
        allmiss = sorted(sub.loc[sub["present"] & (sub["percent_missing"] >= max_missing_pct), "cycle"])
        reasons = []
        if absent:
            reasons.append(f"not collected in cycle(s) {', '.join(absent)}")
        if len(labels) > 1:
            reasons.append(f"question label differs across cycles: {sorted(labels)}")
        if allmiss:
            reasons.append(f"effectively all missing in cycle(s) {', '.join(allmiss)}")
        decisions[v] = {
            "usable": not reasons,
            "reason": "; ".join(reasons) if reasons else "present with identical label in all train/validation/test cycles",
            "max_percent_missing": float(sub["percent_missing"].max()),
            "min_percent_missing": float(sub["percent_missing"].min()),
        }
    return decisions


def main() -> None:
    if not COHORT_PATH.exists():
        raise SystemExit(f"Cohort not found: {COHORT_PATH}. Run ml/preprocessing/extract_nhanes_zip.py first.")
    cohort = pd.read_csv(COHORT_PATH)
    v2, avail = build_v2_cohort(cohort)
    if COHORT_V2_PATH.resolve() == COHORT_PATH.resolve():
        raise AssertionError("Refusing to overwrite the original cohort")
    v2.to_csv(COHORT_V2_PATH, index=False)
    avail.to_csv(AVAILABILITY_PATH, index=False)
    print(f"Wrote {COHORT_V2_PATH.name}: {v2.shape[0]} rows x {v2.shape[1]} cols, positives {int(v2['target_stone'].sum())}")
    print("Dropped KIQ029. History availability (percent missing after decoding):")
    print(avail.drop(columns=["label"]).to_string(index=False))
    decisions = select_history_features(
        avail,
        ["2007-2008", "2009-2010", "2011-2012", "2013-2014"], ["2015-2016"], ["2017-2018"],
    )
    for v, d in decisions.items():
        print(f"  {v}: {'USE' if d['usable'] else 'EXCLUDE'} - {d['reason']}")


if __name__ == "__main__":
    main()
