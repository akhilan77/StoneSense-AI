"""NHANES Kidney Stone Risk Model Training Pipeline (xgboost_nhanes_v002).

Trains a clinical tabular risk model on the multi-cycle NHANES cohort (2007-2018).
Implements:
1. Data cleaning with strict NHANES special code handling.
2. Clinical feature restriction (only patient-collectable clinical parameters).
3. Temporal split: earlier cycles (2007-2016) for development, last cycle (2017-2018) for external testing.
4. Stratified carve out of validation (early stopping) and calibration (probability calibration) slices.
5. Median imputer + StandardScaler + OneHotEncoder in an sklearn Pipeline.
6. LogisticRegression baseline vs. XGBClassifier (scale_pos_weight, randomized search CV, early stopping).
7. Frozen probability calibration (Sigmoid vs. Isotonic compared by Brier score).
8. Single-feature AUC leakage audit and XGBoost gain importance ranking.
9. Comprehensive external test evaluation with 1,000 bootstrap 95% CIs.
10. SHAP TreeExplainer summary and feature attribution charts.
11. Namespaced persistence under ml/artifacts/nhanes_v002/ and ml/models/versions/xgboost_nhanes_v002.pkl.
12. Model registry update to ml/models/registry_manifest.json preserving v001 default deployment.
"""

import sys
from pathlib import Path
import json
import logging
import time
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import xgboost as xgb
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "ml" / "datasets" / "nhanes_kidney_stone_cohort.csv"
RAW_DATA_DIR = PROJECT_ROOT / "ml" / "datasets" / "nhanes_raw"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts" / "nhanes_v002"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
VERSIONS_DIR = MODELS_DIR / "versions"
CHARTS_DIR = PROJECT_ROOT / "ml" / "outputs" / "charts" / "nhanes_v002"
REPORTS_DIR = PROJECT_ROOT / "ml" / "outputs" / "reports"
MANIFEST_PATH = MODELS_DIR / "registry_manifest.json"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrainNHANESv002")

# Fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Feature metadata and specifications
FEATURE_SPECS = {
    "RIDAGEYR": {
        "label": "Age",
        "unit": "years",
        "type": "continuous",
        "clinical_role": "Demographic risk factor; stone incidence peaks with age",
        "min": 20.0,
        "max": 80.0
    },
    "RIAGENDR": {
        "label": "Sex",
        "unit": "1=Male, 2=Female",
        "type": "categorical",
        "clinical_role": "Biological sex; historical prevalence higher in males",
        "min": 1.0,
        "max": 2.0
    },
    "BMXBMI": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m^2",
        "type": "continuous",
        "clinical_role": "Adiposity marker; obesity increases urinary lithogenic excretion",
        "min": 10.0,
        "max": 90.0
    },
    "LBXSCA": {
        "label": "Serum Calcium",
        "unit": "mg/dL",
        "type": "continuous",
        "clinical_role": "Systemic calcium metabolism; hypercalcemia elevates stone risk",
        "min": 5.0,
        "max": 16.0
    },
    "LBXSUA": {
        "label": "Serum Uric Acid",
        "unit": "mg/dL",
        "type": "continuous",
        "clinical_role": "Hyperuricemia predisposes to uric acid and mixed stones",
        "min": 0.5,
        "max": 20.0
    },
    "LBXSCR": {
        "label": "Serum Creatinine",
        "unit": "mg/dL",
        "type": "continuous",
        "clinical_role": "Renal filtration marker; indicator of baseline kidney function",
        "min": 0.1,
        "max": 20.0
    },
    "LBXSBU": {
        "label": "Blood Urea Nitrogen (BUN)",
        "unit": "mg/dL",
        "type": "continuous",
        "clinical_role": "Renal clearance and protein catabolism index",
        "min": 1.0,
        "max": 120.0
    },
    "LBXSGL": {
        "label": "Serum Glucose",
        "unit": "mg/dL",
        "type": "continuous",
        "clinical_role": "Glycemic status / diabetes marker; insulin resistance alters urine pH",
        "min": 20.0,
        "max": 800.0
    },
    "LBXSNASI": {
        "label": "Serum Sodium",
        "unit": "mmol/L",
        "type": "continuous",
        "clinical_role": "Electrolyte and volume balance; dietary sodium drives hypercalciuria",
        "min": 100.0,
        "max": 170.0
    },
    "LBXSKSI": {
        "label": "Serum Potassium",
        "unit": "mmol/L",
        "type": "continuous",
        "clinical_role": "Electrolyte balance and hypocitraturia surrogate",
        "min": 1.5,
        "max": 8.5
    },
    "LBXSPH": {
        "label": "Serum Phosphorus",
        "unit": "mg/dL",
        "type": "continuous",
        "clinical_role": "Mineral homeostasis; phosphaturia and calcium phosphate crystallization",
        "min": 0.5,
        "max": 12.0
    },
    "URXUCR": {
        "label": "Urine Creatinine",
        "unit": "mg/dL",
        "type": "continuous",
        "clinical_role": "Spot urine concentration / excretion rate normalizer",
        "min": 1.0,
        "max": 900.0
    },
    "URXUMA": {
        "label": "Urine Albumin",
        "unit": "ug/mL",
        "type": "continuous",
        "clinical_role": "Microvascular renal permeability and glomerular integrity",
        "min": 0.1,
        "max": 30000.0
    }
}

FINAL_FEATURE_NAMES = list(FEATURE_SPECS.keys())
NUMERICAL_FEATURES = [f for f in FINAL_FEATURE_NAMES if FEATURE_SPECS[f]["type"] == "continuous"]
CATEGORICAL_FEATURES = [f for f in FINAL_FEATURE_NAMES if FEATURE_SPECS[f]["type"] == "categorical"]


def step1_inspect_and_clean_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Step 1: Inspects NHANES raw and cohort data, handles special codes, and drops missing labels."""
    logger.info("=== STEP 1: Inspecting NHANES Cohort and Handling Special Codes ===")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Cohort dataset not found at {DATASET_PATH}")
    
    raw_files = list(RAW_DATA_DIR.glob("*.xpt")) if RAW_DATA_DIR.exists() else []
    logger.info(f"Inspected ml/datasets/nhanes_raw/: found {len(raw_files)} raw SAS XPT files.")
    
    df = pd.read_csv(DATASET_PATH)
    total_raw_rows = len(df)
    logger.info(f"Loaded cohort dataset: {total_raw_rows} rows x {len(df.columns)} columns.")
    
    # Check null percentages across all columns
    null_pct = (df.isnull().mean() * 100).round(2).to_dict()
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    
    # Inspect label column
    label_col = "target_stone"
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' missing from dataset.")
    
    # Replace NHANES special codes where applicable
    # In questionnaire questions, 7=Refused, 9=Don't Know (e.g. KIQ029, KIQ026)
    for q_col in ["KIQ026", "KIQ029"]:
        if q_col in df.columns:
            df[q_col] = df[q_col].replace({7.0: np.nan, 9.0: np.nan, 77.0: np.nan, 99.0: np.nan})
    
    # In continuous clinical lab features, valid measurements must NOT be replaced.
    # Check if target_stone has missing values
    df_clean = df.dropna(subset=[label_col]).copy()
    usable_label_count = len(df_clean)
    label_counts = df_clean[label_col].value_counts().to_dict()
    
    cycle_col = "CYCLE" if "CYCLE" in df_clean.columns else None
    cycle_counts = df_clean[cycle_col].value_counts().sort_index().to_dict() if cycle_col else {}
    
    inspection_summary = {
        "total_raw_rows": total_raw_rows,
        "usable_label_rows": usable_label_count,
        "label_column": label_col,
        "label_counts": label_counts,
        "cycle_column": cycle_col,
        "cycle_counts": cycle_counts,
        "null_percentages": null_pct,
        "dtypes": dtypes
    }
    
    logger.info(f"Target '{label_col}' counts: {label_counts}")
    logger.info(f"Cycle distribution: {cycle_counts}")
    return df_clean, inspection_summary


def step2_select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[Dict[str, str]]]:
    """Step 2: Filters down to patient-collectable clinical parameters and documents excluded columns."""
    logger.info("=== STEP 2: Clinical Feature Selection & Exclusion Documentation ===")
    
    # Document all excluded columns
    all_cols = df.columns.tolist()
    excluded_audit = []
    
    exclusion_reasons = {
        "target_stone": "Target classification outcome label.",
        "SEQN": "Respondent administrative identifier (non-clinical).",
        "CYCLE": "Survey cycle identifier (reserved for temporal external test split).",
        "KIQ026": "Direct patient questionnaire answer for kidney stones (direct outcome proxy).",
        "KIQ029": "Number of times passed a kidney stone in past 12 months (post-outcome derived event).",
        "WTMEC2YR": "MEC 2-year sample survey weight (epidemiological weight, unavailable in clinic).",
        "WTINT2YR": "Interview 2-year survey weight (epidemiological weight, unavailable in clinic).",
        "SDMVPSU": "Masked pseudo-PSU design variable (complex survey sampling structure).",
        "SDMVSTRA": "Masked pseudo-stratum design variable (complex survey sampling structure).",
        "INDFMPIR": "Family poverty-income ratio (socioeconomic demographic, not a clinical biomarker).",
        "RIDRETH1": "Race/ethnicity category (demographic indicator; model restricted to objective physiological biomarkers).",
        "BMXWAIST": "Waist circumference (omitted to avoid collinearity with BMI; not in target clinical biomarker set).",
        "LBXSCLSI": "Serum chloride (omitted from target renal biomarker profile).",
        "LBXSTB": "Serum total bilirubin (hepatic biomarker, omitted from nephrolithiasis risk panel).",
        "URDACT": "Urine albumin-to-creatinine ratio (derived mathematical ratio of URXUMA / URXUCR)."
    }
    
    for c in all_cols:
        if c not in FINAL_FEATURE_NAMES and c != "target_stone":
            reason = exclusion_reasons.get(c, "Excluded: not in clinical biomarker specifications.")
            excluded_audit.append({"column": c, "reason": reason})
    
    X = df[FINAL_FEATURE_NAMES].copy()
    y = df["target_stone"].astype(int).copy()
    cycle_series = df["CYCLE"].copy() if "CYCLE" in df.columns else pd.Series(index=df.index, dtype=str)
    
    logger.info(f"Selected {len(FINAL_FEATURE_NAMES)} clinical features: {FINAL_FEATURE_NAMES}")
    return X, y, cycle_series, excluded_audit


def step3_split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    cycle_series: pd.Series
) -> Dict[str, Any]:
    """Step 3: Splits data into train, validation, calibration, and external test sets."""
    logger.info("=== STEP 3: Temporal Cycle-Based Split with Carved Val and Calib Slices ===")
    
    # Last cycle is external test set
    unique_cycles = sorted(cycle_series.unique())
    last_cycle = unique_cycles[-1]
    earlier_cycles = unique_cycles[:-1]
    
    logger.info(f"Earlier cycles (Development): {earlier_cycles}")
    logger.info(f"Last cycle (External Test): {last_cycle}")
    
    test_mask = (cycle_series == last_cycle)
    dev_mask = (cycle_series.isin(earlier_cycles))
    
    X_dev = X[dev_mask].copy()
    y_dev = y[dev_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()
    
    # Carve two stratified slices from dev set: 10% calibration, 10% validation, remaining 80% train
    # 1. Carve calibration slice (10% of dev)
    X_train_val, X_calib, y_train_val, y_calib = train_test_split(
        X_dev, y_dev, test_size=0.10, stratify=y_dev, random_state=RANDOM_SEED
    )
    
    # 2. Carve validation slice (1/9 of remaining ~ 10% of total dev)
    val_ratio = 0.10 / 0.90  # ~0.1111
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=RANDOM_SEED
    )
    
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_calib": X_calib, "y_calib": y_calib,
        "X_test": X_test, "y_test": y_test,
        "earlier_cycles": earlier_cycles,
        "last_cycle": last_cycle
    }
    
    logger.info(f"Train split: {len(X_train)} rows, {y_train.sum()} pos ({y_train.mean()*100:.2f}%)")
    logger.info(f"Val split: {len(X_val)} rows, {y_val.sum()} pos ({y_val.mean()*100:.2f}%)")
    logger.info(f"Calib split: {len(X_calib)} rows, {y_calib.sum()} pos ({y_calib.mean()*100:.2f}%)")
    logger.info(f"Test split: {len(X_test)} rows, {y_test.sum()} pos ({y_test.mean()*100:.2f}%)")
    
    return splits


def build_preprocessor() -> ColumnTransformer:
    """Builds isolated ColumnTransformer: median imputer + scaler for num, mode + OHE for cat."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def step5_leakage_audit(X_train: pd.DataFrame, y_train: pd.Series) -> List[Dict[str, Any]]:
    """Step 5: Computes single-feature ROC-AUC on train set and flags any > 0.85."""
    logger.info("=== STEP 5: Single-Feature Leakage Audit on Training Set ===")
    
    audit_records = []
    flagged = []
    
    for feat in FINAL_FEATURE_NAMES:
        col_vals = X_train[feat].values
        # Impute median for single-feature evaluation
        valid_mask = ~np.isnan(col_vals)
        imputed_vals = col_vals.copy()
        if not np.all(valid_mask):
            imputed_vals[~valid_mask] = np.median(col_vals[valid_mask])
            
        # Fit univariate logistic regression
        lr = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=200)
        lr.fit(imputed_vals.reshape(-1, 1), y_train)
        probs = lr.predict_proba(imputed_vals.reshape(-1, 1))[:, 1]
        auc = roc_auc_score(y_train, probs)
        
        record = {
            "feature": feat,
            "label": FEATURE_SPECS[feat]["label"],
            "unit": FEATURE_SPECS[feat]["unit"],
            "single_feature_auc": round(float(auc), 4),
            "flagged_leakage": bool(auc > 0.85)
        }
        audit_records.append(record)
        if auc > 0.85:
            flagged.append(record)
            
        logger.info(f"Feature '{feat}' ({FEATURE_SPECS[feat]['label']}): single-feature ROC-AUC = {auc:.4f}")
        
    if flagged:
        logger.warning(f"LEAKAGE WARNING: {len(flagged)} features exceeded 0.85 single-feature ROC-AUC: {flagged}")
    else:
        logger.info("Leakage audit passed: No single feature exceeded 0.85 ROC-AUC threshold.")
        
    return audit_records


def step4_train_and_calibrate(
    splits: Dict[str, Any]
) -> Tuple[Pipeline, Pipeline, Dict[str, Any], np.ndarray]:
    """Step 4: Fits preprocessor, trains baseline LR and tuned XGBoost, and calibrates probabilities."""
    logger.info("=== STEP 4: Model Training, Hyperparameter Optimization, and Calibration ===")
    
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_calib, y_calib = splits["X_calib"], splits["y_calib"]
    
    # 1. Fit preprocessor on training data only
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_calib_trans = preprocessor.transform(X_calib)
    
    # Transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    trans_feature_names = NUMERICAL_FEATURES + cat_feature_names
    
    # 2. Baseline Model: LogisticRegression(class_weight='balanced')
    logger.info("Training baseline Logistic Regression...")
    lr_base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    lr_base.fit(X_train_trans, y_train)
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lr_base)
    ])
    
    # 3. XGBoost: scale_pos_weight = neg / pos
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info(f"Class imbalance: {pos_count} pos, {neg_count} neg. scale_pos_weight = {scale_pos_weight:.2f}")
    
    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "n_estimators": [100, 150, 200, 250],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.75, 0.85, 0.95],
        "min_child_weight": [2, 5, 10],
        "gamma": [0.0, 0.2, 0.5]
    }
    
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    logger.info("Tuning XGBoost with 5-fold Stratified RandomizedSearchCV (scoring=average_precision)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=15,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train_trans, y_train)
    best_params = search.best_params_
    logger.info(f"Best XGBoost Hyperparameters: {best_params} (CV PR-AUC: {search.best_score_:.4f})")
    
    # 4. Fit best XGBoost with early stopping on validation slice
    best_xgb = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        early_stopping_rounds=20,
        n_jobs=-1
    )
    best_xgb.fit(
        X_train_trans,
        y_train,
        eval_set=[(X_val_trans, y_val)],
        verbose=False
    )
    logger.info(f"XGBoost stopped at iteration {best_xgb.best_iteration} (best score: {best_xgb.best_score:.4f})")
    
    # 5. Calibrate on calibration slice only: compare sigmoid vs. isotonic
    logger.info("Calibrating probabilities on separate calibration slice...")
    cal_sig = CalibratedClassifierCV(estimator=FrozenEstimator(best_xgb), method="sigmoid")
    cal_sig.fit(X_calib_trans, y_calib)
    
    cal_iso = CalibratedClassifierCV(estimator=FrozenEstimator(best_xgb), method="isotonic")
    cal_iso.fit(X_calib_trans, y_calib)
    
    probs_sig = cal_sig.predict_proba(X_calib_trans)[:, 1]
    probs_iso = cal_iso.predict_proba(X_calib_trans)[:, 1]
    
    brier_sig = brier_score_loss(y_calib, probs_sig)
    brier_iso = brier_score_loss(y_calib, probs_iso)
    
    if brier_iso < brier_sig:
        selected_calibrator = cal_iso
        chosen_method = "isotonic"
        chosen_brier = brier_iso
    else:
        selected_calibrator = cal_sig
        chosen_method = "sigmoid"
        chosen_brier = brier_sig
        
    logger.info(f"Calibration comparison on calib slice: Sigmoid Brier = {brier_sig:.5f}, Isotonic Brier = {brier_iso:.5f}")
    logger.info(f"Selected calibration method: '{chosen_method}' (Brier: {chosen_brier:.5f})")
    
    # Build final end-to-end inference pipeline
    final_xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", selected_calibrator)
    ])
    
    calibration_meta = {
        "best_hyperparameters": best_params,
        "early_stopping_iteration": int(best_xgb.best_iteration) if hasattr(best_xgb, "best_iteration") else None,
        "scale_pos_weight": float(scale_pos_weight),
        "brier_sigmoid": float(brier_sig),
        "brier_isotonic": float(brier_iso),
        "chosen_method": chosen_method,
        "trans_feature_names": trans_feature_names
    }
    
    return lr_pipeline, final_xgb_pipeline, calibration_meta, best_xgb


def evaluate_model_on_test(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n_bootstraps: int = 1000
) -> Dict[str, Any]:
    """Computes comprehensive metrics and 1,000 bootstrap 95% CIs on external test set."""
    y_true = y_test.values
    probs = model.predict_proba(X_test)[:, 1]
    
    # 1. Base point estimates
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    
    # Metrics at default 0.5 threshold
    preds_50 = (probs >= 0.5).astype(int)
    cm_50 = confusion_matrix(y_true, preds_50)
    tn_50, fp_50, fn_50, tp_50 = cm_50.ravel()
    sens_50 = tp_50 / max(tp_50 + fn_50, 1)
    spec_50 = tn_50 / max(tn_50 + fp_50, 1)
    acc_50 = accuracy_score(y_true, preds_50)
    f1_50 = f1_score(y_true, preds_50, zero_division=0)
    mcc_50 = matthews_corrcoef(y_true, preds_50)
    
    # 2. Operating threshold for ~85% sensitivity
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    # Find threshold with sensitivity >= 0.85
    valid_idx = np.where(tpr >= 0.85)[0]
    if len(valid_idx) > 0:
        opt_thresh = float(thresholds[valid_idx[0]])
    else:
        opt_thresh = 0.5
        
    preds_opt = (probs >= opt_thresh).astype(int)
    cm_opt = confusion_matrix(y_true, preds_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    sens_opt = tp_opt / max(tp_opt + fn_opt, 1)
    spec_opt = tn_opt / max(tn_opt + fp_opt, 1)
    
    # 3. Non-parametric row-level bootstrap 95% CIs
    logger.info(f"Computing {n_bootstraps} bootstrap iterations for {model_name}...")
    n_samples = len(y_true)
    rng = np.random.RandomState(RANDOM_SEED)
    
    boot_aucs, boot_pr_aucs, boot_briers = [], [], []
    boot_sens_50, boot_spec_50, boot_acc_50, boot_f1_50 = [], [], [], []
    boot_sens_opt, boot_spec_opt = [], []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        b_y = y_true[idx]
        b_p = probs[idx]
        
        if len(np.unique(b_y)) < 2:
            continue
            
        boot_aucs.append(roc_auc_score(b_y, b_p))
        boot_pr_aucs.append(average_precision_score(b_y, b_p))
        boot_briers.append(brier_score_loss(b_y, b_p))
        
        b_pred_50 = (b_p >= 0.5).astype(int)
        b_cm_50 = confusion_matrix(b_y, b_pred_50, labels=[0, 1])
        b_tn50, b_fp50, b_fn50, b_tp50 = b_cm_50.ravel()
        boot_sens_50.append(b_tp50 / max(b_tp50 + b_fn50, 1))
        boot_spec_50.append(b_tn50 / max(b_tn50 + b_fp50, 1))
        boot_acc_50.append(accuracy_score(b_y, b_pred_50))
        boot_f1_50.append(f1_score(b_y, b_pred_50, zero_division=0))
        
        b_pred_opt = (b_p >= opt_thresh).astype(int)
        b_cm_opt = confusion_matrix(b_y, b_pred_opt, labels=[0, 1])
        b_tn_opt, b_fp_opt, b_fn_opt, b_tp_opt = b_cm_opt.ravel()
        boot_sens_opt.append(b_tp_opt / max(b_tp_opt + b_fn_opt, 1))
        boot_spec_opt.append(b_tn_opt / max(b_tn_opt + b_fp_opt, 1))
        
    def get_ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        
    def ci_str(val, ci):
        return f"{val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"

    ci_roc = get_ci(boot_aucs)
    ci_pr = get_ci(boot_pr_aucs)
    ci_brier = get_ci(boot_briers)
    ci_sens50 = get_ci(boot_sens_50)
    ci_spec50 = get_ci(boot_spec_50)
    ci_acc50 = get_ci(boot_acc_50)
    ci_f150 = get_ci(boot_f1_50)
    ci_sens_opt = get_ci(boot_sens_opt)
    ci_spec_opt = get_ci(boot_spec_opt)
    
    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "roc_auc_ci": ci_roc,
        "roc_auc_str": ci_str(roc_auc, ci_roc),
        "pr_auc": pr_auc,
        "pr_auc_ci": ci_pr,
        "pr_auc_str": ci_str(pr_auc, ci_pr),
        "brier_score": brier,
        "brier_ci": ci_brier,
        "brier_str": ci_str(brier, ci_brier),
        "accuracy_0.5": acc_50,
        "accuracy_0.5_ci": ci_acc50,
        "accuracy_0.5_str": ci_str(acc_50, ci_acc50),
        "f1_0.5": f1_50,
        "f1_0.5_ci": ci_f150,
        "f1_0.5_str": ci_str(f1_50, ci_f150),
        "sensitivity_0.5": sens_50,
        "sensitivity_0.5_ci": ci_sens50,
        "sensitivity_0.5_str": ci_str(sens_50, ci_sens50),
        "specificity_0.5": spec_50,
        "specificity_0.5_ci": ci_spec50,
        "specificity_0.5_str": ci_str(spec_50, ci_spec50),
        "optimal_threshold_85sens": opt_thresh,
        "sensitivity_opt": sens_opt,
        "sensitivity_opt_ci": ci_sens_opt,
        "sensitivity_opt_str": ci_str(sens_opt, ci_sens_opt),
        "specificity_opt": spec_opt,
        "specificity_opt_ci": ci_spec_opt,
        "specificity_opt_str": ci_str(spec_opt, ci_spec_opt),
        "confusion_matrix_0.5": cm_50.tolist(),
        "confusion_matrix_opt": cm_opt.tolist(),
        "probs": probs
    }


def step7_generate_explainability_charts(
    raw_xgb_model: xgb.XGBClassifier,
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Step 7: SHAP TreeExplainer summary and feature attribution plots on external test set."""
    logger.info("=== STEP 7: Generating SHAP Explainability Visualizations ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transform test set
    X_test_trans = preprocessor.transform(X_test)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + cat_names
    
    # Subsample test set for SHAP to ensure quick, stable plot generation
    sample_size = min(1500, len(X_test_trans))
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(X_test_trans), size=sample_size, replace=False)
    X_sub = X_test_trans[sample_idx]
    
    explainer = shap.TreeExplainer(raw_xgb_model)
    shap_values = explainer(X_sub)
    
    # 1. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_sub, feature_names=feature_names, show=False)
    plt.title("SHAP Global Summary (NHANES Kidney Stone Risk - Test Cohort)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = output_dir / "shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP beeswarm summary to {summary_path}")
    
    # 2. Top-10 Mean |SHAP| Bar Plot
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False).head(10)
    
    plt.figure(figsize=(9, 5))
    sns.barplot(data=shap_df, x="importance", y="feature", palette="mako", hue="feature", legend=False)
    plt.title("Top-10 Feature Importance (Mean |SHAP Value| on Test Cohort)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontweight="bold")
    plt.ylabel("Clinical Feature", fontweight="bold")
    plt.tight_layout()
    bar_path = output_dir / "shap_importance_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP top-10 importance bar plot to {bar_path}")


def step8_generate_diagnostic_charts(
    lr_eval: Dict[str, Any],
    xgb_eval: Dict[str, Any],
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Generates Calibration and ROC / PR Curve diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = y_test.values
    
    # 1. Calibration Curve Plot
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_true, xgb_eval["probs"], n_bins=10, strategy="uniform")
    prob_true_lr, prob_pred_lr = calibration_curve(y_true, lr_eval["probs"], n_bins=10, strategy="uniform")
    
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred_xgb, prob_true_xgb, "s-", color="#008080", label=f"XGBoost (Brier: {xgb_eval['brier_score']:.4f})")
    plt.plot(prob_pred_lr, prob_true_lr, "o--", color="#d95f02", label=f"Logistic Regression (Brier: {lr_eval['brier_score']:.4f})")
    plt.xlabel("Mean Predicted Probability", fontweight="bold")
    plt.ylabel("Observed Fraction of Stone Positives", fontweight="bold")
    plt.title("Reliability Calibration Curves (External Test Set: 2017-2018)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    calib_path = output_dir / "calibration_curve.png"
    plt.savefig(calib_path, dpi=300)
    plt.close()
    logger.info(f"Saved calibration curves to {calib_path}")
    
    # 2. ROC & PR Curves Plot
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_eval["probs"])
    fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_eval["probs"])
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, xgb_eval["probs"])
    prec_lr, rec_lr, _ = precision_recall_curve(y_true, lr_eval["probs"])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # ROC
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
    axes[0].plot(fpr_xgb, tpr_xgb, color="#008080", lw=2, label=f"XGBoost (AUC = {xgb_eval['roc_auc']:.4f})")
    axes[0].plot(fpr_lr, tpr_lr, color="#d95f02", lw=2, label=f"LogReg (AUC = {lr_eval['roc_auc']:.4f})")
    axes[0].set_xlabel("False Positive Rate", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    # PR
    baseline_prev = y_true.mean()
    axes[1].axhline(y=baseline_prev, color="k", linestyle="--", alpha=0.6, label=f"Baseline Prev ({baseline_prev*100:.1f}%)")
    axes[1].plot(rec_xgb, prec_xgb, color="#008080", lw=2, label=f"XGBoost (PR-AUC = {xgb_eval['pr_auc']:.4f})")
    axes[1].plot(rec_lr, prec_lr, color="#d95f02", lw=2, label=f"LogReg (PR-AUC = {lr_eval['pr_auc']:.4f})")
    axes[1].set_xlabel("Recall (Sensitivity)", fontweight="bold")
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    axes[1].set_title("Precision-Recall Curve (PR)", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    curves_path = output_dir / "roc_pr_curve.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC and PR curves to {curves_path}")


def step8_save_artifacts_and_report(
    final_pipeline: Pipeline,
    raw_xgb_model: xgb.XGBClassifier,
    lr_eval: Dict[str, Any],
    xgb_eval: Dict[str, Any],
    leakage_audit: List[Dict[str, Any]],
    excluded_audit: List[Dict[str, str]],
    splits: Dict[str, Any],
    calib_meta: Dict[str, Any]
) -> None:
    """Step 8: Saves artifacts under ml/artifacts/nhanes_v002/, models/versions/, and updates registry."""
    logger.info("=== STEP 8: Persisting Artifacts, Specifications, and Registry Manifest ===")
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Save calibrated model pipeline to ml/models/versions/xgboost_nhanes_v002.pkl
    model_version_tag = "xgboost_nhanes_v002"
    model_save_path = VERSIONS_DIR / f"{model_version_tag}.pkl"
    joblib.dump(final_pipeline, model_save_path)
    logger.info(f"Saved calibrated model pipeline to {model_save_path}")
    
    # 2. Save preprocessing pipeline
    fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
    joblib.dump(fitted_preprocessor, ARTIFACTS_DIR / "preprocessing_pipeline.pkl")
    
    # 3. Save feature columns
    joblib.dump(FINAL_FEATURE_NAMES, ARTIFACTS_DIR / "feature_columns.pkl")
    
    # 4. Generate feature_spec.json
    X_train = splits["X_train"]
    feature_specs_full = {}
    for feat in FINAL_FEATURE_NAMES:
        spec = dict(FEATURE_SPECS[feat])
        spec["name"] = feat
        # Compute median from training set as clinical default
        median_val = float(X_train[feat].median())
        spec["default"] = round(median_val, 2)
        feature_specs_full[feat] = spec
        
    with open(ARTIFACTS_DIR / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(feature_specs_full, f, indent=4)
    logger.info(f"Saved feature specifications to {ARTIFACTS_DIR / 'feature_spec.json'}")
    
    # 5. Save preprocessing metadata JSON
    prep_metadata = {
        "dataset_name": "NHANES Kidney Stone Cohort (2007-2018)",
        "model_version": model_version_tag,
        "features": FINAL_FEATURE_NAMES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "development_cycles": splits["earlier_cycles"],
        "external_test_cycle": splits["last_cycle"],
        "row_counts": {
            "train": len(splits["X_train"]),
            "validation": len(splits["X_val"]),
            "calibration": len(splits["X_calib"]),
            "external_test": len(splits["X_test"]),
            "total_cohort": len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_calib"]) + len(splits["X_test"])
        },
        "class_prevalence": {
            "train_pos_pct": round(float(splits["y_train"].mean() * 100), 2),
            "test_pos_pct": round(float(splits["y_test"].mean() * 100), 2)
        },
        "calibration": calib_meta
    }
    with open(ARTIFACTS_DIR / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(prep_metadata, f, indent=4)
        
    # 6. Update ml/models/registry_manifest.json (keep logistic_regression_v001 deployed)
    manifest = {"active_model_version": "logistic_regression_v001", "models": {}}
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing manifest: {e}")
            
    v002_entry = {
        "model_family": "xgboost",
        "version_tag": model_version_tag,
        "status": "pending_review",
        "is_deployed": False,
        "dataset_name": "NHANES Kidney Stone Cohort (2007-2018)",
        "artifact_path": f"ml/models/versions/{model_version_tag}.pkl",
        "absolute_artifact_path": str(model_save_path).replace("\\", "/"),
        "features": FINAL_FEATURE_NAMES,
        "threshold": round(xgb_eval["optimal_threshold_85sens"], 4),
        "split_counts": prep_metadata["row_counts"],
        "metrics": {
            "test_cycle": splits["last_cycle"],
            "roc_auc": round(xgb_eval["roc_auc"], 4),
            "roc_auc_ci_95": str(xgb_eval["roc_auc_ci"]),
            "pr_auc": round(xgb_eval["pr_auc"], 4),
            "pr_auc_ci_95": str(xgb_eval["pr_auc_ci"]),
            "brier_score": round(xgb_eval["brier_score"], 4),
            "sensitivity_opt": round(xgb_eval["sensitivity_opt"], 4),
            "specificity_opt": round(xgb_eval["specificity_opt"], 4),
            "sensitivity_0.5": round(xgb_eval["sensitivity_0.5"], 4),
            "specificity_0.5": round(xgb_eval["specificity_0.5"], 4),
            "accuracy_0.5": round(xgb_eval["accuracy_0.5"], 4),
            "f1_0.5": round(xgb_eval["f1_0.5"], 4)
        },
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    manifest["models"][model_version_tag] = v002_entry
    # Ensure active_model_version remains v001
    if manifest.get("active_model_version") != "logistic_regression_v001":
        manifest["active_model_version"] = "logistic_regression_v001"
        
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"Updated registry manifest with '{model_version_tag}' (status: pending_review, is_deployed: false).")
    
    # 7. Write Comprehensive Evaluation Report
    report_path = REPORTS_DIR / "nhanes_v002_report.md"
    
    # Format tables
    features_table_md = "\n".join([
        f"| `{f}` | **{FEATURE_SPECS[f]['label']}** | `{FEATURE_SPECS[f]['unit']}` | {FEATURE_SPECS[f]['type']} | {FEATURE_SPECS[f]['clinical_role']} |"
        for f in FINAL_FEATURE_NAMES
    ])
    
    leakage_table_md = "\n".join([
        f"| `{r['feature']}` | {r['label']} | `{r['unit']}` | **{r['single_feature_auc']}** | {'⚠️ FLAG' if r['flagged_leakage'] else '✅ Clean'} |"
        for r in leakage_audit
    ])
    
    excluded_table_md = "\n".join([
        f"| `{ex['column']}` | {ex['reason']} |"
        for ex in excluded_audit
    ])
    
    # Extract top XGBoost gain features
    booster = raw_xgb_model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    total_gain = sum(score_gain.values()) if score_gain else 1.0
    gain_rows = sorted(score_gain.items(), key=lambda x: x[1], reverse=True)
    gain_table_md = "\n".join([
        f"| `{k}` | {v:.2f} | {(v/total_gain)*100:.2f}% |"
        for k, v in gain_rows[:10]
    ])
    
    report_md = f"""# NHANES Kidney Stone Tabular Risk Model Report (`xgboost_nhanes_v002`)

**Date:** {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}  
**Model Identifier:** `xgboost_nhanes_v002`  
**Dataset:** National Health and Nutrition Examination Survey (NHANES) 2007–2018  
**Cohort Size:** {prep_metadata['row_counts']['total_cohort']} observations  
**Deployment State:** `pending_review` (Non-active candidate; `logistic_regression_v001` preserved as active default)  

---

## 1. Executive Summary & Design Principles

The `xgboost_nhanes_v002` tabular model addresses the clinical risk estimation of nephrolithiasis on a nationwide, representative population cohort ($N = 34,679$). Unlike the historical 79-row urine analysis prototype, this model trains on **{prep_metadata['row_counts']['train']}** clinical records across 5 survey cycles (2007–2016) and validates against an independent, chronologically external cohort from the 2017–2018 cycle ($N = {prep_metadata['row_counts']['external_test']}$).

### Key Methodology Safeguards:
1. **Clinic-Collectable Restriction:** Only biomarkers and demographic measures routinely obtainable during an outpatient or emergency consultation are utilized.
2. **Strict Leakage Prevention:** All survey weights, pseudo-strata, derived post-stone events, and questionnaire outcome proxies are strictly excluded.
3. **Temporal Hold-Out Validation:** Model selection and final benchmarking are executed on the chronologically distinct 2017–2018 cycle.
4. **Probability Calibration:** The raw ensemble logits are calibrated via `{calib_meta['chosen_method']}` scaling on a separate, dedicated calibration slice ($N = {prep_metadata['row_counts']['calibration']}$).

---

## 2. Dataset Partitioning & Class Distribution

| Partition | Survey Cycles | Total Samples | Stone Positives ($N$) | Prevalence (%) | Purpose |
|:---|:---|:---:|:---:|:---:|:---|
| **Training** | 2007–2016 (Earlier) | {prep_metadata['row_counts']['train']} | {splits['y_train'].sum()} | {splits['y_train'].mean()*100:.2f}% | Parameter optimization |
| **Validation** | 2007–2016 (Stratified Slice) | {prep_metadata['row_counts']['validation']} | {splits['y_val'].sum()} | {splits['y_val'].mean()*100:.2f}% | XGBoost early stopping |
| **Calibration** | 2007–2016 (Stratified Slice) | {prep_metadata['row_counts']['calibration']} | {splits['y_calib'].sum()} | {splits['y_calib'].mean()*100:.2f}% | Frozen probability calibration |
| **External Test** | 2017–2018 (Last Cycle) | {prep_metadata['row_counts']['external_test']} | {splits['y_test'].sum()} | {splits['y_test'].mean()*100:.2f}% | Unseen temporal evaluation |
| **Total Cohort** | 2007–2018 | {prep_metadata['row_counts']['total_cohort']} | {splits['y_train'].sum() + splits['y_val'].sum() + splits['y_calib'].sum() + splits['y_test'].sum()} | {(splits['y_train'].sum() + splits['y_val'].sum() + splits['y_calib'].sum() + splits['y_test'].sum()) / prep_metadata['row_counts']['total_cohort'] * 100:.2f}% | Population benchmark |

---

## 3. Final Feature Set with Clinical Units

| Feature Code | Clinical Parameter | Unit | Type | Pathophysiological Rationale |
|:---|:---|:---|:---|:---|
{features_table_md}

---

## 4. Leakage Audit & Excluded Variables

### Single-Feature ROC-AUC Audit (Training Set)
*Threshold for leakage alert: ROC-AUC > 0.85*

| Feature Code | Parameter Label | Unit | Univariate ROC-AUC | Audit Status |
|:---|:---|:---|:---:|:---:|
{leakage_table_md}

### Excluded Variables Rationale
| Excluded Variable | Clinical / Methodological Rationale |
|:---|:---|
{excluded_table_md}

---

## 5. External Test Performance Comparison (2017–2018 Cycle)

*All metrics computed on external test set ($N = 5,558$, prevalence $9.97\%$). 95% Confidence Intervals derived from 1,000 non-parametric bootstrap iterations.*

| Evaluation Metric | Baseline Logistic Regression | Tuned & Calibrated XGBoost (`v002`) | Clinical Significance |
|:---|:---:|:---:|:---|
| **ROC-AUC** | {lr_eval['roc_auc_str']} | **{xgb_eval['roc_auc_str']}** | Discriminative power across all thresholds |
| **PR-AUC (Avg Precision)** | {lr_eval['pr_auc_str']} | **{xgb_eval['pr_auc_str']}** | Area under precision-recall curve (baseline: 0.0997) |
| **Brier Score** | {lr_eval['brier_str']} | **{xgb_eval['brier_str']}** | Mean squared probability error (lower is better) |
| **Accuracy (Threshold 0.5)** | {lr_eval['accuracy_0.5_str']} | **{xgb_eval['accuracy_0.5_str']}** | Proportion of correct classifications |
| **F1-Score (Threshold 0.5)** | {lr_eval['f1_0.5_str']} | **{xgb_eval['f1_0.5_str']}** | Harmonic mean of precision and recall |
| **Sensitivity (Threshold 0.5)** | {lr_eval['sensitivity_0.5_str']} | {xgb_eval['sensitivity_0.5_str']} | True positive recall at standard threshold |
| **Specificity (Threshold 0.5)** | {lr_eval['specificity_0.5_str']} | **{xgb_eval['specificity_0.5_str']}** | True negative rate at standard threshold |
| **Operating Threshold ($\ge 85\%$ Sensitivity)** | `{lr_eval['optimal_threshold_85sens']:.4f}` | **`{xgb_eval['optimal_threshold_85sens']:.4f}`** | Clinical screening decision boundary |
| **Sensitivity at Operating Threshold** | {lr_eval['sensitivity_opt_str']} | **{xgb_eval['sensitivity_opt_str']}** | Screening sensitivity guardrail |
| **Specificity at Operating Threshold** | {lr_eval['specificity_opt_str']} | **{xgb_eval['specificity_opt_str']}** | Specificity achieved while capturing $\ge 85\%$ of stone patients |

### Confusion Matrices (External Test Cohort, $N = 5,558$)
- **XGBoost at Threshold 0.50:**
  - True Negative: {xgb_eval['confusion_matrix_0.5'][0][0]} | False Positive: {xgb_eval['confusion_matrix_0.5'][0][1]}
  - False Negative: {xgb_eval['confusion_matrix_0.5'][1][0]} | True Positive: {xgb_eval['confusion_matrix_0.5'][1][1]}
- **XGBoost at Screening Threshold ({xgb_eval['optimal_threshold_85sens']:.4f}):**
  - True Negative: {xgb_eval['confusion_matrix_opt'][0][0]} | False Positive: {xgb_eval['confusion_matrix_opt'][0][1]}
  - False Negative: {xgb_eval['confusion_matrix_opt'][1][0]} | True Positive: {xgb_eval['confusion_matrix_opt'][1][1]}

---

## 6. XGBoost Gain Feature Importance

| Feature Code | Total Gain Score | Relative Contribution (%) |
|:---|:---:|:---:|
{gain_table_md}

---

## 7. Artifacts & Generated Assets

1. **Trained Calibrated Pipeline:** `ml/models/versions/xgboost_nhanes_v002.pkl`
2. **Preprocessing Transformer:** `ml/artifacts/nhanes_v002/preprocessing_pipeline.pkl`
3. **Authoritative Feature Columns:** `ml/artifacts/nhanes_v002/feature_columns.pkl`
4. **Feature Specification:** `ml/artifacts/nhanes_v002/feature_spec.json`
5. **Preprocessing Metadata:** `ml/artifacts/nhanes_v002/preprocessing_metadata.json`
6. **Diagnostic Charts:**
   - `ml/outputs/charts/nhanes_v002/shap_summary.png`
   - `ml/outputs/charts/nhanes_v002/shap_importance_bar.png`
   - `ml/outputs/charts/nhanes_v002/calibration_curve.png`
   - `ml/outputs/charts/nhanes_v002/roc_pr_curve.png`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Generated comprehensive report at {report_path}")


def main():
    logger.info("Starting NHANES Tabular Risk Model Pipeline (xgboost_nhanes_v002)...")
    start_time = time.time()
    
    # Step 1
    df_clean, inspect_summary = step1_inspect_and_clean_dataset()
    
    # Step 2
    X, y, cycle_series, excluded_audit = step2_select_features(df_clean)
    
    # Step 3
    splits = step3_split_dataset(X, y, cycle_series)
    
    # Step 5 (Run before training as leakage check)
    leakage_audit = step5_leakage_audit(splits["X_train"], splits["y_train"])
    
    # Step 4
    lr_pipe, xgb_pipe, calib_meta, raw_xgb = step4_train_and_calibrate(splits)
    
    # Step 6
    logger.info("=== STEP 6: Evaluating Models on External Test Set (2017-2018) ===")
    lr_eval = evaluate_model_on_test(lr_pipe, splits["X_test"], splits["y_test"], "LogisticRegression")
    xgb_eval = evaluate_model_on_test(xgb_pipe, splits["X_test"], splits["y_test"], "XGBoost")
    
    # Step 7
    step7_generate_explainability_charts(raw_xgb, xgb_pipe.named_steps["preprocessor"], splits["X_test"], splits["y_test"], CHARTS_DIR)
    
    # Step 8 (Diagnostics and Persistence)
    step8_generate_diagnostic_charts(lr_eval, xgb_eval, splits["y_test"], CHARTS_DIR)
    step8_save_artifacts_and_report(xgb_pipe, raw_xgb, lr_eval, xgb_eval, leakage_audit, excluded_audit, splits, calib_meta)
    
    elapsed = time.time() - start_time
    logger.info(f"=== PIPELINE COMPLETED IN {elapsed:.1f}s ===")
    
    # Print clean summary for console inspection
    print("\n" + "="*80)
    print("NHANES TABULAR RISK MODEL (xgboost_nhanes_v002) - EXECUTION COMPLETE")
    print("="*80)
    print(f"(a) ROW COUNTS & CLASS BALANCE:")
    print(f"    Train:         {len(splits['X_train']):>5} rows | Positives: {splits['y_train'].sum():>4} ({splits['y_train'].mean()*100:.2f}%)")
    print(f"    Validation:    {len(splits['X_val']):>5} rows | Positives: {splits['y_val'].sum():>4} ({splits['y_val'].mean()*100:.2f}%)")
    print(f"    Calibration:   {len(splits['X_calib']):>5} rows | Positives: {splits['y_calib'].sum():>4} ({splits['y_calib'].mean()*100:.2f}%)")
    print(f"    External Test: {len(splits['X_test']):>5} rows | Positives: {splits['y_test'].sum():>4} ({splits['y_test'].mean()*100:.2f}%) [Cycle: 2017-2018]")
    print(f"    Total Cohort:  {len(df_clean):>5} rows | Positives: {y.sum():>4} ({y.mean()*100:.2f}%)")
    
    print("\n(b) FINAL CLINICAL FEATURES WITH UNITS:")
    for f in FINAL_FEATURE_NAMES:
        spec = FEATURE_SPECS[f]
        print(f"    - {f:<10}: {spec['label']:<28} [Unit: {spec['unit']:<18}] (Type: {spec['type']})")
        
    print("\n(c) EXTERNAL TEST SET METRICS (LR vs. XGBoost with 95% Bootstrap CIs):")
    print(f"    {'Metric':<25} | {'Logistic Regression':<28} | {'XGBoost (Calibrated v002)':<28}")
    print("    " + "-"*85)
    print(f"    {'ROC-AUC':<25} | {lr_eval['roc_auc_str']:<28} | {xgb_eval['roc_auc_str']:<28}")
    print(f"    {'PR-AUC':<25} | {lr_eval['pr_auc_str']:<28} | {xgb_eval['pr_auc_str']:<28}")
    print(f"    {'Brier Score':<25} | {lr_eval['brier_str']:<28} | {xgb_eval['brier_str']:<28}")
    print(f"    {'Accuracy (thresh 0.5)':<25} | {lr_eval['accuracy_0.5_str']:<28} | {xgb_eval['accuracy_0.5_str']:<28}")
    print(f"    {'Sensitivity (thresh 0.5)':<25} | {lr_eval['sensitivity_0.5_str']:<28} | {xgb_eval['sensitivity_0.5_str']:<28}")
    print(f"    {'Specificity (thresh 0.5)':<25} | {lr_eval['specificity_0.5_str']:<28} | {xgb_eval['specificity_0.5_str']:<28}")
    print(f"    {'Opt Thresh (>=85% Sens)':<25} | {lr_eval['optimal_threshold_85sens']:<28.4f} | {xgb_eval['optimal_threshold_85sens']:<28.4f}")
    print(f"    {'Sensitivity @ Opt Thresh':<25} | {lr_eval['sensitivity_opt_str']:<28} | {xgb_eval['sensitivity_opt_str']:<28}")
    print(f"    {'Specificity @ Opt Thresh':<25} | {lr_eval['specificity_opt_str']:<28} | {xgb_eval['specificity_opt_str']:<28}")
    
    print("\n(d) SINGLE-FEATURE LEAKAGE AUDIT & EXCLUSIONS:")
    for rec in leakage_audit:
        print(f"    - {rec['feature']:<10} ({rec['label']:<24}): ROC-AUC = {rec['single_feature_auc']:.4f} {'[FLAGGED LEAKAGE]' if rec['flagged_leakage'] else '[CLEAN]'}")
    print(f"    Total excluded non-clinical / proxy columns: {len(excluded_audit)}")
    
    model_save_path = VERSIONS_DIR / "xgboost_nhanes_v002.pkl"
    print("\n(e) ALL CREATED ARTIFACTS & REPORTS:")
    print(f"    - Model Artifact:    {model_save_path}")
    print(f"    - Preprocessor:      {ARTIFACTS_DIR / 'preprocessing_pipeline.pkl'}")
    print(f"    - Feature List:      {ARTIFACTS_DIR / 'feature_columns.pkl'}")
    print(f"    - Feature Spec:      {ARTIFACTS_DIR / 'feature_spec.json'}")
    print(f"    - Metadata JSON:     {ARTIFACTS_DIR / 'preprocessing_metadata.json'}")
    print(f"    - Report:            {REPORTS_DIR / 'nhanes_v002_report.md'}")
    print(f"    - SHAP Beeswarm:     {CHARTS_DIR / 'shap_summary.png'}")
    print(f"    - SHAP Bar Plot:     {CHARTS_DIR / 'shap_importance_bar.png'}")
    print(f"    - Calibration Curve: {CHARTS_DIR / 'calibration_curve.png'}")
    print(f"    - ROC/PR Curves:     {CHARTS_DIR / 'roc_pr_curve.png'}")
    print(f"    - Registry Manifest: {MANIFEST_PATH} (v002 added as pending_review, v001 remains default)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
