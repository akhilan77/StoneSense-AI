# StoneSense-AI Model Cards

**Date:** October 2026  
**Document Status:** Audited Benchmark Release  
**License:** MIT (Application Code) / Respective Upstream Weights Licenses  

> [!CAUTION]
> ### ⚠️ EXPLICIT CLINICAL NON-INTENDED USE NOTICE
> **The models described in this document are academic research prototypes designed exclusively for proof-of-concept computational evaluation.**  
> **THEY ARE NOT INTENDED FOR CLINICAL USE, MEDICAL DIAGNOSIS, PATIENT TRIAGE, OR SURGICAL TREATMENT PLANNING.**  
> Neither StoneSense-AI nor its contributors have obtained clearance or approval from the U.S. Food and Drug Administration (FDA), European Medicines Agency (EMA), CDSCO, or any other global regulatory authority for diagnostic applications.

---

## Model Card 1: Tabular Kidney Stone Risk Model

### 1.1 Overview & Intended Use
- **Model Families Evaluated:** Logistic Regression, Random Forest, XGBoost.
- **Active Deployed Version:** `logistic_regression_v001` (Selected via 1-Standard-Error rule).
- **Secondary Version:** `xgboost_v001`.
- **Intended Purpose:** Educational and research assessment of patient risk probability derived from standard urine biochemistry panels.
- **Non-Intended Purpose:** Primary diagnosis of nephrolithiasis or replacement of imaging-based clinical assessment.
- **Related:** the NHANES clinical-profile models (EBM, CatBoost, XGBoost, LightGBM; `nhanes_v003b`, pending review) are documented in Model Card 5.

### 1.2 Training Data & Features
- **Data Source:** Historical clinical bench test cohort (Andrews & Lyne urine biochemistry dataset).
- **Cohort Size:** 79 unique observations following strict duplicate filtering (cleaned from 80-row raw cohort).
- **Input Features (6 continuous numeric biomarkers):**
  1. `gravity`: Specific gravity of urine (g/cm³).
  2. `ph`: Urine pH level.
  3. `osmo`: Urine osmolarity (mOsm).
  4. `cond`: Urine conductivity (mMho).
  5. `urea`: Urine urea concentration (millimoles/L).
  6. `calc`: Calcium concentration (millimoles/L).
- **Target:** Binary risk classification (`0`: Low Risk, `1`: High Risk).

### 1.3 Validation Protocol & Split Method
- **Protocol:** 5-fold cross-validation $\times$ 10 repeats (**50 total cross-validation folds**), stratified by target class.
- **Hyperparameter Tuning:** Nested `GridSearchCV` inside outer cross-validation folds (preventing test fold hyperparameter leakage).
- **Confidence Intervals:** Non-parametric row-level bootstrap (1,000 iterations).
- **Significance Testing:** Permutation testing across 1,000 random label permutations.

### 1.4 Verified Metrics (Source: `ml/models/comparison_results.csv`)

| Metric | Logistic Regression (Deployed Baseline) | XGBoost (v001) | Random Forest |
| :--- | :---: | :---: | :---: |
| **Mean ROC-AUC (95% CI)** | **0.8387** `[0.7327, 0.9176]` | 0.8076 `[0.7080, 0.9156]` | 0.8175 `[0.7122, 0.9191]` |
| **Mean Macro F1 (95% CI)** | **0.6937** `[0.5333, 0.7937]` | 0.7169 `[0.5862, 0.8308]` | 0.7067 `[0.5778, 0.8302]` |
| **Mean Accuracy (95% CI)** | **74.67%** `[0.6329, 0.8231]` | 77.14% `[0.6835, 0.8608]` | 75.62% `[0.6835, 0.8608]` |
| **Mean MCC (95% CI)** | **0.4926** `[0.2399, 0.6397]` | 0.5505 `[0.3420, 0.7138]` | 0.5152 `[0.3398, 0.7123]` |
| **Sensitivity (95% CI)** | **`[0.4749, 0.8000]`** | `[0.5152, 0.8333]` | `[0.5143, 0.8276]` |
| **Specificity (95% CI)** | **`[0.6842, 0.9091]`** | `[0.7347, 0.9412]` | `[0.7292, 0.9388]` |
| **Permutation $p$-value** | **0.001** | 0.001 | 0.001 |
| **Model Selection Decision** | **1-SE Rule Winner** | Higher Complexity | Higher Complexity |

### 1.5 Explainability Method
- Dynamic local attribution using **SHAP**:
  - Linear models utilize `shap.LinearExplainer` with precomputed feature background distribution.
  - Tree ensembles utilize `shap.TreeExplainer`.
  - Returns signed attribution values indicating whether a specific biomarker increased or decreased predicted risk.

### 1.6 Known Limitations
- Extreme sample size constraint ($N=79$). Large confidence intervals.
- Lack of demographic diversity (age, biological sex, BMI, and geographic origin are not represented).
- Historical "91.67% accuracy" claim was measured on an unstratified 12-sample test subset and is discredited.

---

## Model Card 2: Axial CT Kidney Classification Model (ResNet18)

### 2.1 Overview & Intended Use
- **Architecture:** ResNet18 (He et al., 2016) transfer-learning convolutional network initialized from `ResNet18_Weights.IMAGENET1K_V1`.
- **Intended Purpose:** Proof-of-concept visual saliency mapping and multiclass axial CT slice categorization.
- **Classes (4 classes):** `Cyst`, `Normal`, `Stone`, `Tumor`.
- **Output:** Softmax probability distribution and predicted class label.

### 2.2 Training Data & Split Methodology
- **Data Source:** Public axial CT slice collection (12,446 total images).
- **Crucial Split Protocol:** **Perceptual Cluster Grouping ($pHash \le 2$)**:
  - The raw dataset lacked patient identifiers. A random slice-level shuffle suffered from 227 exact duplicate pairs and 703 perceptual slice clusters crossing splits.
  - Slices were grouped into 979 indivisible cluster units using 64-bit DCT perceptual hash Hamming distance ($\le 2$).
  - **Train Split (69.97%):** 8,708 slices (699 clusters).
  - **Validation Split (15.02%):** 1,869 slices (138 clusters).
  - **Held-Out Test Split (15.02%):** 1,869 slices (142 clusters).

### 2.3 Verified Metrics (Source: `dl/models/ct/resnet18/metrics.json` & `docs/RESNET18_AUDIT_REPORT.md`)

| Metric | Audited Leak-Free Held-Out Test Set | Historical Leaky Baseline (Discredited) |
| :--- | :---: | :---: |
| **Slice Accuracy** | **86.94%** (1,625 / 1,869) | 99.63% (1,864 / 1,871) |
| **Slice Macro F1** | **82.51%** | 99.47% |
| **Slice Weighted F1** | **87.37%** | 99.62% |
| **Cyst Recall** | **70.74%** (394 / 557) | 99.82% |
| **Normal Recall** | **98.69%** (752 / 762) | 100.0% |
| **Stone Recall (Diagnostic Sensitivity)**| **65.70%** (136 / 207) | 97.12% |
| **Tumor Recall (Diagnostic Sensitivity)**| **100.0%** (343 / 343) | 99.71% |
| **Scan / Cluster Accuracy (Majority Vote)**| **96.48%** (137 / 142) | N/A (Unclustered) |
| **Scan / Cluster Macro F1** | **95.47%** | N/A |

### 2.4 Explainability Method
- **Standard Grad-CAM** (Selvaraju et al., 2017) applied to `model.layer4[-1]`.
- Gradients of the target predicted class score are global average pooled to compute spatial channel weights, generating a normalized saliency heatmap blended with the input CT slice.

### 2.5 Known Limitations
- Modest single-slice sensitivity for kidney stones (65.70%) due to small stone volume across non-target slices. Cluster-level majority voting substantially mitigates this (96.48% scan accuracy).
- Lacks patient metadata verification: pHash grouping is a computational proxy for volume grouping, but true cross-hospital patient isolation cannot be proven without original clinical DICOM headers.

---

## Model Card 3: Rule-Based Evidence Concordance Engine

### 3.1 Overview & Architecture
- **Algorithm:** Deterministic rule-based agreement engine implemented in [`backend/app/services/concordance_service.py`](backend/app/services/concordance_service.py).
- **Inputs:** Independent tabular urine biomarker risk prediction and axial CT imaging prediction.
- **Outcome Codes:**
  1. `CONCORDANT_STONE`: Both streams present positive evidence for nephrolithiasis.
  2. `CONCORDANT_NO_STONE_EVIDENCE`: Neither stream shows stone evidence.
  3. `DISCORDANT_CT_POSITIVE`: CT indicates stone while urine chemistry indicates low risk.
  4. `DISCORDANT_BIOMARKER_POSITIVE`: Urine chemistry indicates high risk while CT imaging shows normal parenchyma.
  5. `INCONCLUSIVE_OOD`: CT image was flagged by out-of-distribution / uncertainty gate or quality filter.
  6. `NON_STONE_PATHOLOGY_PRESENT`: CT imaging detected Cyst or Tumor, superseding non-stone classification.

### 3.2 Critical Clinical Safety Disclosures & Limitations
> [!WARNING]
> **CONCORDANT_NO_STONE_EVIDENCE IS NOT A CLINICAL CLEARANCE:**
> 1. **Diagnostic Sensitivity Deficit:** The audited axial CT model misses a substantial share of stones (**Slice Stone Recall is only 65.70%**; Scan-level Stone Recall is 80.95%). Approximately 1 in 3 positive stone slices will produce a false negative if assessed individually.
> 2. **Extreme Sample Constraint:** The tabular urine risk model was trained on a very small historical cohort of only **79 patients** (95% CI sensitivity: `[0.4749, 0.8000]`).
> 3. **Unpaired Cohorts:** The tabular dataset and axial CT dataset are **not patient-paired**. There are zero ground-truth linked records across streams; agreement is an artificial heuristic decision-support cross-check, not a joint multimodal multimodal model.
> 4. **Uncalibrated Thresholds:** Concordance decision boundaries (e.g. Risk $\ge 0.5$, CT Stone probability $\ge 0.5$) are uncalibrated default thresholds, not clinically validated cutoffs.

---

## Model Card 4: Supplementary & Experimental Architectures

The repository includes modular model wrappers in [`dl/models/wrappers/`](dl/models/wrappers/) evaluated on the identical held-out grouped test set:

| Architecture | Paradigm | Test Accuracy | Macro F1 | Stone Recall | Status / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLO26** | Real-time CNN Backbone | **92.19%** | **90.47%** | **94.20%** | Production-ready candidate weights in `dl/models/ct/yolo26/runs/yolo26n_ct/weights/best.pt`. |
| **DINOv3** | Vision Transformer (Self-supervised) | **87.43%** | **82.81%** | **61.84%** | Features extracted from ViT backbone; evaluation completed. |
| **QKNN** | Quantum-Classical Simulator (PennyLane) | **41.95%** | **35.89%** | **25.60%** | Research experiment. PCA dimension reduction (4 qubits) degrades clinical resolution; unsuitable for deployment. |

---

## Model Card 5: NHANES Clinical-Profile Risk Models (`nhanes_v003b`: EBM, CatBoost, XGBoost, LightGBM)

> **Status: `pending_review`, not deployed.** The active tabular model remains `logistic_regression_v001`; the earlier `*_nhanes_v003` registry entries are kept. Every number below was written by the executed notebooks `ml/notebooks/nhanes_v003/00-05` (run with `python ml/training/run_nhanes_notebooks.py`) into `ml/outputs/metrics/nhanes_v003/*.json` and `ml/outputs/reports/nhanes_v003_report.md`; the previous run's files are kept in `ml/outputs/metrics/nhanes_v003/_prev_v003/`.

### 5.1 Overview & Intended Use
- **What it is:** four independently trained tabular models (Explainable Boosting Machine, CatBoost, XGBoost, LightGBM) that estimate the probability that a US adult *reports ever having had a kidney stone*, from routine blood chemistry, urine, body measures and four self-reported history items (plus optional race / income). A logistic regression on the same data is the reference baseline (not registered).
- **Intended use:** research and teaching: comparing model families on one frozen split, and showing four models side by side in the StoneSense risk page (`POST /api/v1/predict/risk/compare`).
- **Not intended for:** diagnosis, triage, or any clinical decision. The label is self-reported and the signal is small (see 5.6).
- **What changed in `nhanes_v003b`:** (1) LightGBM no longer stops after a handful of boosting rounds (5.4); (2) five class-imbalance strategies are compared honestly in every notebook (5.5); (3) two operating points are stored for every model: `accuracy` (highest sensitivity with validation accuracy >= 72%, to reach > 70% test accuracy) and `balanced` (maximum balanced accuracy); (4) every notebook shows the full metric set, not only AUC.

### 5.2 Data
- **Source:** NHANES 2007-2018 (cycles E-J), 34,679 participants with a Yes/No answer to `KIQ026` ("ever had kidney stones"); 3,234 positives. The cohort CSV was re-verified against the raw `KIQ_U` files (row and positive counts match in all six cycles).
- **Target:** `target_stone` (`KIQ026` = Yes).
- **Base features (17):** age, sex, BMI, serum calcium / uric acid / creatinine / BUN / glucose / sodium / potassium / phosphorus, urine creatinine and albumin (the 13 `xgboost_nhanes_v002` inputs) plus waist circumference, serum chloride, serum total bilirubin and the urine albumin/creatinine ratio (derived server-side).
- **History features used (4):** `BPQ020`, `BPQ080`, `DIQ010` (borderline = 0.5), `MCQ080`. `MCQ520` and `MCQ550` are excluded (not collected in every cycle); no cycle is ever imputed wholesale.
- **Sensitive features:** race/ethnicity (`RIDRETH1`) and income ratio (`INDFMPIR`) are optional inputs; see 5.5 for what they do to the results.
- **Never used as features:** `SEQN, CYCLE, target_stone, KIQ026, KIQ029` (dropped from the v2 cohort), `WTMEC2YR, WTINT2YR, SDMVPSU, SDMVSTRA`. A single-feature AUC audit found no feature above the 0.85 flag threshold.

### 5.3 Splits and Validation Protocol
Frozen, hash-checked splits **by survey cycle** (hash `a2c3bdb15ffb9f3b`; identical in all five model notebooks; no `SEQN` shared between splits):

| Split | Survey cycles | Rows | Stone positives | Positive rate |
|:---|:---|:---:|:---:|:---:|
| train | 2007-2008, 2009-2010, 2011-2012, 2013-2014 | 23,416 | 2,074 | 8.86% |
| validation | 2015-2016 | 5,705 | 606 | 10.62% |
| test | 2017-2018 | 5,558 | 554 | 9.97% |

Prevalence differs between splits; this is reported, not adjusted. Imputers and scalers are fit on the training cycles only. **Resampling is applied only to training rows, inside the fit step of an `imblearn` Pipeline; the validation and test cycles are never resampled and the test cycle keeps its real ~10% prevalence.** The 2015-2016 validation cycle is split by `SEQN` (stratified) into an **early-stopping half** (2,852 rows, 303 positives: stopping point and hyper-parameter score) and a **calibration / threshold half** (2,853 rows, 303 positives: imbalance-strategy selection, calibration, both operating thresholds and every validation table), so the rows that decide where boosting stops are not the rows that calibrate probabilities and place thresholds. The test cycle is scored **once per frozen model** (a ledger enforces this) with 1,000-sample bootstrap 95% CIs and is never used to choose a threshold, hyper-parameters, a balancing method or features. Leave-one-cycle-out CV over the five train+validation cycles is a stability check.

**Disclosure.** The 2017-2018 test cycle had already been scored once by `nhanes_v003` before this revision was designed. Nothing in `nhanes_v003b` was chosen with test data, but the decision to revise the operating point and the LightGBM settings was informed by seeing the earlier results; the test numbers below are a second look, not a first one.

### 5.4 LightGBM early-stop diagnosis and fix
`nhanes_v003` LightGBM stopped at `best_iteration_` = 3 (registered variant). Notebook 04 reproduces that (3 rounds kept, 153 trained) and tests the suspected causes against per-round train / validation curves (`ml/outputs/charts/nhanes_v003/lightgbm_early_stop_curves.png`):

| Suspected cause | Verdict (computed in the notebook) |
|:---|:---:|
| (a) early stopping used another metric (logloss) instead of AUC | **true** |
| (b) scale_pos_weight inflates probabilities, so UNWEIGHTED validation logloss rises at once | **true** |
| (c) learning rate / leaves too loose (old params reach their logloss minimum much sooner than the new defaults, even unweighted) | false |
| (d) validation cycle differs from the training cycles (logloss minimum comes earlier there than on a random train hold-out) | false |
| (e) stopping_rounds too small (patience 200 would not stop at the same place as 150) | false |
| (f) same validation rows used for early stopping AND calibration / thresholds | **true** |

**Root cause:** the old call passed `eval_metric='auc'`, but LightGBM also evaluates the objective's default `binary_logloss`, and its early-stopping callback stops as soon as *any* metric stops improving. With `scale_pos_weight` the predicted probabilities are inflated, so the *unweighted* validation logloss was best after 3 rounds and then rose, while AUC kept improving until round 159. (The old setup also early-stopped and calibrated on the same validation rows.)

**Fix:** stop on validation AUC only (`first_metric_only=True`, patience 200) while still logging average precision and logloss each round; slow learning rate, shallow trees, large leaves and regularisation; exactly one imbalance strategy (chosen in the strategy comparison) instead of stacking weights and resampling; early-stopping half kept apart from the calibration half. **Before / after `best_iteration_`: 3 -> 190** (rounds trained 390). The same check on the other boosters (registered variant, final tuned models): XGBoost best_iteration 1083 (rounds trained 1183), CatBoost 448 (598), LightGBM 190 (390); none stops suspiciously early. Each notebook asserts `best_iteration > 30` before anything is calibrated or scored on test.

### 5.5 Class-imbalance handling
Compared on the same frozen split in every model notebook: `none`, class weights (`scale_pos_weight` / `sample_weight` / `class_weight`), random oversampling, random undersampling and SMOTENC (the binary and one-hot columns are declared categorical so no invalid in-between values are created). All are calibrated on the real-prevalence validation calibration half (balanced training raises the average predicted risk from ~10% to ~25-46%; calibration brings it back to the real ~10.6%). The strategy is selected on validation only by a pre-declared rule (best post-calibration PR-AUC; strategies within 0.005 PR-AUC tie, then lowest Brier, then simplest). Full metric tables for every strategy on validation and on test are in each notebook.

| Model | Selected | PR-AUC gain over `none` (calibrated, CV) | `none` within tie band | ROC-AUC range over the 5 strategies (validation) | Test ROC-AUC: `none` -> selected |
|:---|:---:|:---:|:---:|:---:|:---:|
| EBM | none | +0.0000 | True | 0.0666 | 0.679 -> 0.679 |
| CatBoost | class_weight | +0.0066 | False | 0.0225 | 0.662 -> 0.674 |
| XGBoost | none | +0.0000 | True | 0.0292 | 0.674 -> 0.674 |
| LightGBM | none | +0.0000 | True | 0.0281 | 0.676 -> 0.676 |
| Logistic regression (reference) | none | +0.0000 | True | 0.0301 | 0.661 -> 0.661 |

**1 of 4 models gained more than the tie band in PR-AUC over "none + calibration + threshold"**; for the others `none` tied or won, so the expectation that balancing mostly shifts the decision threshold and rarely improves ranking is confirmed here, not assumed. With 303 positives in the calibration half, PR-AUC differences of about 0.005 are within sampling noise. Random oversampling never beat the simpler options by more than the tie band, and SMOTENC was clearly worse on ROC-AUC than 'none' for every model.

### 5.6 Verified Test Metrics (held-out 2017-2018, n = 5,558, stone prevalence 10.0%; registered variant `base_history_sensitive`)
**About 90.0% of participants have no stone, so predicting "no stone" for everyone scores 90.0% accuracy.** Two operating points per model, both chosen on the validation calibration half only:

| Model | Operating point | Threshold | Test accuracy [95% CI] | Majority baseline | Sensitivity [95% CI] | Specificity [95% CI] | Balanced acc. [95% CI] | Flagged | ROC-AUC [95% CI] | PR-AUC | Brier |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| EBM | accuracy | 0.1314 | 71.6% [70.4, 72.8] | 90.0% | 50.4% [46.1, 54.6] | 73.9% [72.7, 75.1] | 62.1% [60.0, 64.2] | 28.5% | 0.674 [0.651, 0.697] | 0.192 | 0.0863 |
| EBM | balanced | 0.1100 | 61.5% [60.3, 62.8] | 90.0% | 63.0% [59.0, 66.9] | 61.3% [60.0, 62.7] | 62.2% [60.0, 64.2] | 41.1% | 0.674 [0.651, 0.697] | 0.192 | 0.0863 |
| CatBoost | accuracy | 0.1332 | 72.7% [71.6, 73.9] | 90.0% | 48.2% [44.1, 52.5] | 75.5% [74.3, 76.7] | 61.8% [59.7, 64.0] | 26.9% | 0.675 [0.653, 0.699] | 0.194 | 0.0863 |
| CatBoost | balanced | 0.1190 | 66.1% [64.9, 67.3] | 90.0% | 57.4% [53.2, 61.7] | 67.0% [65.7, 68.4] | 62.2% [60.0, 64.3] | 35.4% | 0.675 [0.653, 0.699] | 0.194 | 0.0863 |
| XGBoost | accuracy | 0.1316 | 72.5% [71.3, 73.6] | 90.0% | 50.0% [45.8, 54.1] | 75.0% [73.8, 76.2] | 62.5% [60.2, 64.6] | 27.5% | 0.677 [0.655, 0.701] | 0.195 | 0.0862 |
| XGBoost | balanced | 0.1138 | 64.1% [62.9, 65.4] | 90.0% | 60.3% [56.3, 64.4] | 64.5% [63.2, 65.8] | 62.4% [60.2, 64.5] | 37.9% | 0.677 [0.655, 0.701] | 0.195 | 0.0862 |
| LightGBM | accuracy | 0.1316 | 72.4% [71.3, 73.5] | 90.0% | 48.4% [44.1, 52.8] | 75.1% [73.9, 76.2] | 61.7% [59.5, 63.9] | 27.3% | 0.675 [0.653, 0.698] | 0.191 | 0.0863 |
| LightGBM | balanced | 0.1171 | 66.1% [64.8, 67.3] | 90.0% | 57.6% [53.6, 61.7] | 67.0% [65.7, 68.4] | 62.3% [60.1, 64.5] | 35.4% | 0.675 [0.653, 0.698] | 0.191 | 0.0863 |
| Logistic regression (reference) | accuracy | 0.1429 | 84.5% [83.5, 85.4] | 90.0% | 23.6% [20.0, 27.1] | 91.2% [90.5, 92.0] | 57.4% [55.6, 59.3] | 10.3% | 0.654 [0.631, 0.679] | 0.167 | 0.0892 |
| Logistic regression (reference) | balanced | 0.1254 | 59.1% [57.8, 60.3] | 90.0% | 63.9% [60.2, 67.9] | 58.6% [57.2, 59.9] | 61.2% [59.3, 63.4] | 43.7% | 0.654 [0.631, 0.679] | 0.167 | 0.0892 |

**Accuracy goal (> 70% on the held-out test cycle), reached only through the validation-side rule** (maximum sensitivity subject to validation accuracy >= 72%):

- EBM: test accuracy 71.6% at sensitivity 50.4% and specificity 73.9% (**above 70%**; validation accuracy 72.03% at the same cut-off).
- CatBoost: test accuracy 72.7% at sensitivity 48.2% and specificity 75.5% (**above 70%**; validation accuracy 72.03% at the same cut-off).
- XGBoost: test accuracy 72.5% at sensitivity 50.0% and specificity 75.0% (**above 70%**; validation accuracy 72.03% at the same cut-off).
- LightGBM: test accuracy 72.4% at sensitivity 48.4% and specificity 75.1% (**above 70%**; validation accuracy 72.03% at the same cut-off).
- Logistic regression (reference): test accuracy 84.5% at sensitivity 23.6% and specificity 91.2% (**above 70%**; validation accuracy 84.12% at the same cut-off).

**Accuracy above 70% is obtained at the cost of sensitivity.** At the accuracy point the four models flag about 27%-28% of people and find 48.2% to 50.4% of true stone cases; the majority baseline (90.0%) is higher than every one of these accuracies, so they are not evidence of skill. ROC-AUC (0.674 to 0.677) and balanced accuracy (62.2% to 62.4% at the balanced point) are the honest yardsticks, are not expected to exceed 70%, and were not forced. All four confidence intervals for ROC-AUC overlap, so no model can be called better than another from this data. (The logistic-regression reference uses isotonic calibration, whose coarse score steps make the accuracy rule overshoot: validation accuracy 84.1%, test accuracy 84.5% but sensitivity only 23.6%.)

Leave-one-cycle-out stability of the operating points (the cut-off of each fold is placed by the fraction of people the validation-chosen point flags; no held-out label is used):

| Model | LOCO accuracy @ accuracy point | LOCO sensitivity @ accuracy point | LOCO accuracy @ balanced point | LOCO sensitivity @ balanced point |
|:---|:---:|:---:|:---:|:---:|
| EBM | 72.7% +/- 0.3 | 49.0% +/- 1.2 | 64.1% +/- 0.3 | 61.7% +/- 1.8 |
| CatBoost | 72.3% +/- 0.3 | 49.7% +/- 1.6 | 66.7% +/- 0.2 | 57.9% +/- 1.6 |
| XGBoost | 72.4% +/- 0.3 | 49.0% +/- 1.7 | 65.0% +/- 0.2 | 60.5% +/- 1.2 |
| LightGBM | 72.6% +/- 0.3 | 48.2% +/- 1.4 | 66.8% +/- 0.2 | 56.5% +/- 1.4 |
| Logistic regression (reference) | 84.6% +/- 0.4 | 22.2% +/- 0.8 | 60.9% +/- 0.3 | 66.0% +/- 2.4 |

### 5.7 Feature-Set Ablation and the Race / Income Question
| Model | Variant | Validation AUC (calibration half) | LOCO AUC (mean +/- sd, 5 cycles) | Test ROC-AUC (95% CI) |
|:---|:---|:---:|:---:|:---:|
| EBM | `base` | 0.650 | 0.654 +/- 0.011 | 0.650 [0.626, 0.673] |
| EBM | `base_history` | 0.663 | 0.661 +/- 0.010 | 0.654 [0.631, 0.676] |
| EBM | `base_history_sensitive` | 0.663 | 0.683 +/- 0.013 | 0.674 [0.651, 0.697] |
| CatBoost | `base` | 0.637 | 0.642 +/- 0.010 | 0.635 [0.611, 0.660] |
| CatBoost | `base_history` | 0.643 | 0.659 +/- 0.012 | 0.650 [0.626, 0.673] |
| CatBoost | `base_history_sensitive` | 0.654 | 0.684 +/- 0.014 | 0.675 [0.653, 0.699] |
| XGBoost | `base` | 0.639 | 0.650 +/- 0.010 | 0.651 [0.628, 0.674] |
| XGBoost | `base_history` | 0.646 | 0.661 +/- 0.009 | 0.656 [0.633, 0.680] |
| XGBoost | `base_history_sensitive` | 0.659 | 0.684 +/- 0.012 | 0.677 [0.655, 0.701] |
| LightGBM | `base` | 0.637 | 0.647 +/- 0.009 | 0.646 [0.623, 0.670] |
| LightGBM | `base_history` | 0.645 | 0.660 +/- 0.008 | 0.652 [0.627, 0.675] |
| LightGBM | `base_history_sensitive` | 0.660 | 0.682 +/- 0.010 | 0.675 [0.653, 0.698] |
| Logistic regression (reference) | `base` | 0.636 | 0.634 +/- 0.015 | 0.623 [0.598, 0.647] |
| Logistic regression (reference) | `base_history` | 0.644 | 0.651 +/- 0.009 | 0.636 [0.613, 0.659] |
| Logistic regression (reference) | `base_history_sensitive` | 0.664 | 0.675 +/- 0.011 | 0.654 [0.631, 0.679] |

- Race/ethnicity and income are ethically sensitive inputs. The pre-set rule in notebook 00 (adopt only if both leave-one-cycle-out AUC and validation AUC improve by at least 0.010) **adopted them** for the registered variant `base_history_sensitive`. In the app they are optional ("prefer not to say" is allowed; blank values are replaced by the training-cohort typical value and are not shown as patient factors). To register the variant without them, run notebooks 00 and 05 with the environment variable `NHANES_PRIMARY_VARIANT=base_history`; every model notebook already trains and reports all three variants.
- Top contributing features on the validation cycle (registered variant):
  - EBM: `RIDRETH1`, `RIDAGEYR`, `BPQ020`, `URXUMA`, `BMXWAIST`
  - CatBoost: `RIDRETH1`, `RIDAGEYR`, `URXUMA`, `BPQ020`, `BMXWAIST`
  - XGBoost: `RIDRETH1`, `RIDAGEYR`, `BPQ020`, `URXUMA`, `BMXWAIST`
  - LightGBM: `RIDAGEYR`, `RIDRETH1`, `URXUMA`, `BPQ020`, `BMXWAIST`

### 5.8 Known Limitations
- **Self-reported outcome** ("ever had a kidney stone"), not imaging- or lab-confirmed, and not "currently".
- **US adults aged 20-80 only** (NHANES); survey weights were not used, so results are not national estimates and may not transfer to other populations or clinical settings.
- **Small signal:** test ROC-AUC 0.674 to 0.677; the four models' confidence intervals overlap.
- Predictors are cross-sectional and measured at the same visit as the self-reported outcome (associations, not causes).
- Tuning, calibration and thresholds use the 2015-2016 validation cycle (stone prevalence 10.6%), higher than the training cycles (8.9%); the accuracy point is chosen on a half of that cycle (2,853 rows), so its test accuracy can drift from the 72% validation target.
- Test metrics describe performance on held-out NHANES data; they are **not** the accuracy of any individual prediction. The accuracy-focused point flags fewer patients as High risk and misses more true cases.
- Not validated prospectively; not approved for clinical use.

### 5.9 Explainability, Compute, Reproducibility
- **Explanations:** SHAP `TreeExplainer` for XGBoost / CatBoost / LightGBM; native additive term scores for the EBM (interaction terms split evenly between their two features). Returned in the standard StoneSense schema (`top_features`, `feature_contributions`, `feature_directions`, `summary`).
- **Compute:** XGBoost (`device="cuda"`) and CatBoost (`task_type="GPU"`) trained on the GPU; LightGBM used the CPU because the installed wheel has no GPU learner; EBM is CPU-only. Serving needs no GPU. Seeds fixed (42). Libraries: Python 3.12.9, scikit-learn 1.9.0, imbalanced-learn 0.14.2, XGBoost 3.4.1, CatBoost 1.2.10, LightGBM 4.7.0, interpret 0.7.8.

| Model | Notebook | Executed (UTC) | Device | Training time (tuning + fits + CV) | Notebook wall time |
|:---|:---|:---:|:---:|:---:|:---:|
| EBM | `01_ebm.ipynb` | 2026-10-15T08:35:14Z | cpu | 291 s | 316 s |
| CatBoost | `02_catboost.ipynb` | 2026-10-15T08:42:06Z | GPU | 380 s | 407 s |
| XGBoost | `03_xgboost.ipynb` | 2026-10-15T08:44:07Z | cuda:0 | 82 s | 108 s |
| LightGBM | `04_lightgbm.ipynb` | 2026-10-15T08:46:54Z | cpu | 115 s | 162 s |
| Logistic regression (reference) | `04b_logistic_regression_baseline.ipynb` | 2026-10-15T08:47:32Z | cpu | 8 s | 33 s |

- **Provenance:** each metrics JSON records the notebook file, execution timestamp, device, training time, library versions and the split hash; notebook 05 loads only these saved artifacts and fails loudly if a JSON is missing, lacks provenance, or its split hash differs. `testing/test_nhanes_notebooks.py` checks the executed `.ipynb` files against the JSONs.
- **Rerun:** `python ml/training/run_nhanes_notebooks.py` (needs `requirements-ml.txt`, which now includes `imbalanced-learn`); see the README section "NHANES v003b notebooks".

---

## 7-Model Accounting & Federated Learning Capabilities Matrix

StoneSense-AI explicitly accounts for seven distinct computational models across deep learning and tabular machine learning:

| # | Model Identifier | Category | Task | Algorithm / Backbone | FedAvg Capable? | Federation Status | Architectural Ineligibility Reason |
| :-: | :--- | :---: | :--- | :--- | :-: | :---: | :--- |
| **1** | `resnet18` | Deep Learning | Axial CT Classification | PyTorch ResNet18 CNN | **Yes** | **READY** | Standard parameter tensor architecture supported by StoneSenseFLClient. |
| **2** | `yolo26` | Deep Learning | Axial CT Classification | Ultralytics YOLO26n-cls | **No** | **UNSUPPORTED** | Proprietary layer blocks and anchor head parameters are incompatible with plain FedAvg. |
| **3** | `dinov3` | Deep Learning | Axial CT Classification | Meta ViT Foundation | **No** | **UNSUPPORTED** | Frozen foundation backbone used for zero-shot embedding extraction; local fine-tuning requires specialized adaptation. |
| **4** | `qknn` | Deep Learning / Quantum | Axial CT Classification | PennyLane 4-Qubit Simulator | **No** | **UNSUPPORTED** | Quantum kernel angle parameters and distance matrices cannot be averaged via classical arithmetic FedAvg. |
| **5** | `logistic_regression` | Machine Learning | Urine Risk Prediction | L2-Regularized Logistic Regression | **No** | **UNSUPPORTED** | Tabular benchmark trained via 50-fold nested CV; requires secure aggregation rather than gradient FedAvg. |
| **6** | `random_forest` | Machine Learning | Urine Risk Prediction | Scikit-Learn RandomForestClassifier | **No** | **UNSUPPORTED** | Decision tree split structures and thresholds cannot be averaged arithmetically across nodes. |
| **7** | `xgboost` | Machine Learning | Urine & NHANES Risk | XGBClassifier Gradient Boosting | **No** | **UNSUPPORTED** | Tree ensembles with histogram-based leaf splits require specialized Federated Tree algorithms (e.g. Federated XGBoost). |

> The NHANES v003 / v003b families added in Model Card 5 (`ebm`, `catboost`, `lightgbm`, and the new `xgboost` bundle) are tree / additive ensembles and, like rows 5-7, are not FedAvg-capable.

---

## Ethical, Regulatory & Clinical Disclosures

1. **Bias & Population Representation**: The training data comes from unspecified public sources without demographic or regional stratification. Diagnostic accuracy may vary significantly across diverse patient populations.
2. **False Negative Risk**: A 65.70% slice-level stone sensitivity indicates that approximately 1 in 3 positive stone slices will be missed if evaluated in isolation without considering contiguous slice volumes.
3. **No Clinical Actionability**: Outputs provided by StoneSense-AI must always be verified by board-certified radiologists and urologists using certified clinical PACS viewers.
