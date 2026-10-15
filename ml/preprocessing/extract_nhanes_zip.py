"""Safely extract the NHANES raw zip into ml/datasets/ and copy the cohort CSV.

Usage (from repo root):
    python ml/preprocessing/extract_nhanes_zip.py [path/to/nhanes_raw.zip]

Rejects absolute paths, drive letters and ".." entries. The original zip is never modified.
Then reports which of the 18 history files (BPQ/DIQ/MCQ, cycles E-J) are missing or not valid XPT.
"""
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "ml" / "datasets"
DEFAULT_ZIP = Path(r"C:\Users\Sanjeev\Downloads\nhanes_raw.zip")
SUFFIXES = list("EFGHIJ")
HISTORY_PREFIXES = ["BPQ", "DIQ", "MCQ"]


def safe_extract(zip_path: Path, dest: Path) -> list:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or (len(name) > 1 and name[1] == ":"):
                raise ValueError(f"Unsafe zip entry rejected: {info.filename!r}")
            target = (dest / Path(*p.parts)).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Zip entry escapes destination: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target)
    return extracted


def check_history_files(raw_dir: Path) -> list:
    problems = []
    for prefix in HISTORY_PREFIXES:
        for s in SUFFIXES:
            f = raw_dir / f"{prefix}_{s}.xpt"
            if not f.exists():
                problems.append(f"MISSING: {f.name}")
                continue
            try:
                df = pd.read_sas(f, format="xport")
                if "SEQN" not in df.columns or len(df) == 0:
                    problems.append(f"INVALID (no SEQN / empty): {f.name}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"INVALID XPT: {f.name} ({exc})")
    return problems


def main() -> None:
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ZIP
    if not zip_path.exists():
        raise SystemExit(f"Zip not found: {zip_path}")
    files = safe_extract(zip_path, DATASETS_DIR)
    print(f"Extracted {len(files)} files into {DATASETS_DIR}")
    src_csv = DATASETS_DIR / "nhanes_raw" / "nhanes_kidney_stone_cohort.csv"
    dst_csv = DATASETS_DIR / "nhanes_kidney_stone_cohort.csv"
    if src_csv.exists():
        shutil.copyfile(src_csv, dst_csv)
        print(f"Copied cohort CSV -> {dst_csv}")
    problems = check_history_files(DATASETS_DIR / "nhanes_raw")
    if problems:
        print("History file problems:")
        for p in problems:
            print("  ", p)
    else:
        print("All 18 history files (BPQ/DIQ/MCQ E-J) present and valid XPT.")


if __name__ == "__main__":
    main()
