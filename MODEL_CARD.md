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

## Model Card 3: Supplementary & Experimental Architectures

The repository includes modular model wrappers in [`dl/models/wrappers/`](dl/models/wrappers/) evaluated on the identical held-out grouped test set:

| Architecture | Paradigm | Test Accuracy | Macro F1 | Stone Recall | Status / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLO26** | Real-time CNN Backbone | **92.19%** | **90.47%** | **94.20%** | Production-ready candidate weights in `dl/models/ct/yolo26/runs/yolo26n_ct/weights/best.pt`. |
| **DINOv3** | Vision Transformer (Self-supervised) | **87.43%** | **82.81%** | **61.84%** | Features extracted from ViT backbone; evaluation completed. |
| **QKNN** | Quantum-Classical Simulator (PennyLane) | **41.95%** | **35.89%** | **25.60%** | Research experiment. PCA dimension reduction (4 qubits) degrades clinical resolution; unsuitable for deployment. |

---

## Ethical, Regulatory & Clinical Disclosures

1. **Bias & Population Representation**: The training data comes from unspecified public sources without demographic or regional stratification. Diagnostic accuracy may vary significantly across diverse patient populations.
2. **False Negative Risk**: A 65.70% slice-level stone sensitivity indicates that approximately 1 in 3 positive stone slices will be missed if evaluated in isolation without considering contiguous slice volumes.
3. **No Clinical Actionability**: Outputs provided by StoneSense-AI must always be verified by board-certified radiologists and urologists using certified clinical PACS viewers.
