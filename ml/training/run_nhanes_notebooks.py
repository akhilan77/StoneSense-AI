"""Execute the NHANES v003 / v003b notebooks in order with ``jupyter nbconvert --execute`` (fresh kernel per notebook,
top to bottom, outputs saved in place). This script only EXECUTES notebooks: it trains, tunes and evaluates nothing itself.

Usage (from the repo root):
    python ml/training/run_nhanes_notebooks.py                 # real run: executes 00..05 in place (real outputs saved)
    python ml/training/run_nhanes_notebooks.py --dry-run       # development run: the TEST CYCLE IS NOT TOUCHED,
                                                               # outputs go to ml/outputs/_dryrun, notebooks are not modified
    python ml/training/run_nhanes_notebooks.py --only 00 01    # subset (prefix match on the notebook file name)

Order: 00 data/splits -> 01 EBM -> 02 CatBoost -> 03 XGBoost -> 04 LightGBM -> 04b logistic regression -> 05 compare/register.
Each model notebook scores the held-out test cycle once per frozen model; the result is recorded in a ledger
(ml/outputs/metrics/nhanes_v003/test_ledger.json) and re-running an unchanged notebook reuses it. Models are registered under
the tag ``<model>_nhanes_v003b`` (the earlier ``*_nhanes_v003`` ledger entries and registry entries are kept).
Requires: pip install -r requirements-ml.txt
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = PROJECT_ROOT / "ml" / "notebooks" / "nhanes_v003"
NOTEBOOKS = [
    "00_data_and_splits.ipynb",
    "01_ebm.ipynb",
    "02_catboost.ipynb",
    "03_xgboost.ipynb",
    "04_lightgbm.ipynb",
    "04b_logistic_regression_baseline.ipynb",
    "05_compare_and_register.ipynb",
]


def run(dry_run: bool, only: list, timeout: int) -> int:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["NHANES_DRY_RUN"] = "1" if dry_run else "0"
    out_dir = PROJECT_ROOT / "ml" / "outputs" / ("_dryrun" if dry_run else "metrics/nhanes_v003")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [n for n in NOTEBOOKS if not only or any(n.startswith(o) for o in only)]
    log = []
    for name in selected:
        path = NOTEBOOK_DIR / name
        if not path.exists():
            print(f"[missing] {path}")
            return 2
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
               "--ExecutePreprocessor.kernel_name=python3", f"--ExecutePreprocessor.timeout={timeout}"]
        if dry_run:
            cmd += ["--output-dir", str(PROJECT_ROOT / "ml" / "outputs" / "_dryrun" / "notebooks")]
        else:
            cmd += ["--inplace"]
        cmd.append(str(path))
        print(f"=== {name} ({'dry run' if dry_run else 'real run'}) ===", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(NOTEBOOK_DIR), env=env)
        dt = round(time.perf_counter() - t0, 1)
        log.append({"notebook": name, "seconds": dt, "return_code": proc.returncode})
        print(f"--- {name}: {'OK' if proc.returncode == 0 else 'FAILED'} in {dt}s", flush=True)
        if proc.returncode != 0:
            break
    (out_dir / "notebook_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    return 0 if log and all(r["return_code"] == 0 for r in log) and len(log) == len(selected) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="never touch the test cycle; write to ml/outputs/_dryrun")
    ap.add_argument("--only", nargs="*", default=[], help="notebook name prefixes, e.g. 00 01")
    ap.add_argument("--timeout", type=int, default=7200, help="per-notebook timeout in seconds")
    args = ap.parse_args()
    sys.exit(run(args.dry_run, args.only, args.timeout))


if __name__ == "__main__":
    main()
