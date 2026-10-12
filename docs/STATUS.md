# StoneSense-AI System Status & Feature Verification Matrix

**Audit Date:** October 12, 2026  
**Auditor:** Senior Engineering Code Audit  
**Target Environment:** Windows 11 / Python 3.12 / React 18  

---

## 1. Executive Status Summary

This document establishes the verified runtime reality of the **StoneSense-AI** repository. Every claim below has been corroborated through direct source inspection, local automated test execution, static type analysis, or database state examination.

- **Backend Status:** Operational (FastAPI 0.115+, 52 endpoints registered).
- **Backend Test Suite:** **106 passed, 1 skipped, 0 failed** (Total: 107 items, 72% code coverage across `backend/app`).
- **Frontend Status:** Operational (React 18.3 + Vite 5.4 + Tailwind CSS 3.4). `npx tsc --noEmit` exits cleanly (code 0) and `npm run build` generates production assets cleanly in 11.33s.
- **Active ML Model:** `logistic_regression_v001` (Deployed 1-Standard-Error rule winner).
- **Active DL Model:** `resnet18_centralized_v1` / `resnet18_grouped_audit_v1` (Evaluated on grouped leak-free test split).
- **Federated Learning Rounds:** 29 completed live rounds persisted in `stonesense.db`.

---

## 2. Complete Feature Status Matrix

| Module | Feature | Status | Evidence & File Links | Audit Notes |
| :--- | :--- | :---: | :--- | :--- |
| **ML Tabular** | Nested Repeated CV Pipeline | **DONE** | [`ml/training/train_risk_model.py`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/training/train_risk_model.py#L50-L150) | 5 folds $\times$ 10 repeats (50 folds) with row-level bootstrap and permutation significance tests. |
| **ML Tabular** | Model Selection (1-SE Rule) | **DONE** | [`ml/training/evaluate.py`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/training/evaluate.py#L180-L240), [`comparison_results.csv`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/models/comparison_results.csv) | Selects Logistic Regression over XGBoost and Random Forest due to higher mean ROC-AUC (0.8387) and simpler complexity. |
| **ML Tabular** | Model Version Registry | **DONE** | [`ml/risk_models/registry.py`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/risk_models/registry.py), [`registry_manifest.json`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/models/registry_manifest.json) | Manages versioned checkpoints (`logistic_regression_v001.pkl`, `xgboost_v001.pkl`). |
| **ML Tabular** | Preprocessing & Scaling Pipeline | **DONE** | [`ml/artifacts/preprocessing_pipeline.pkl`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/artifacts/preprocessing_pipeline.pkl), [`scaler.pkl`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/artifacts/scaler.pkl) | Authoritative feature names in `feature_columns.pkl`. |
| **XAI (ML)** | Dynamic SHAP Explainer | **DONE** | [`ml/explainability/shap_explainer.py`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/explainability/shap_explainer.py#L79-L90), [`explainability_service.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/explainability_service.py#L80-L92) | Uses `shap.LinearExplainer` for linear models and `shap.TreeExplainer` for tree models; prevents TreeExplainer crash on Logistic Regression. |
| **XAI (ML)** | Local SHAP Exposure via API/UI | **DONE** | [`predict_router.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/predict_router.py#L100-L110), [`PatientForm.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/components/PatientForm.tsx#L305-L330) | Feature attributions and contribution directions returned in JSON and rendered in UI. |
| **DL CT** | ResNet18 Training & Checkpoints | **DONE** | [`dl/training/train_resnet18.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/training/train_resnet18.py), [`kidney_resnet18.pth`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/models/ct/resnet18/kidney_resnet18.pth) | 4-class classification (`Cyst`, `Normal`, `Stone`, `Tumor`). Checkpoints saved in `dl/models/ct/resnet18/`. |
| **DL CT** | Leakage-Controlled Grouped Split | **DONE** | [`dl/training/train_resnet18_audit.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/training/train_resnet18_audit.py), [`metrics.json`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/models/ct/resnet18/metrics.json) | 64-bit DCT perceptual hash grouping ($pHash \le 2$). Confirms historical 99.6% was inflated by 227 duplicate pairs; honest test accuracy is 86.94%. |
| **DL CT** | Multi-Model Architecture Wrappers | **PARTIAL** | [`dl/models/wrappers/`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/models/wrappers/) | ResNet18 (86.94% acc), YOLO26 (92.19% acc), and DINOv3 (87.43% acc) work. QKNN is experimental simulator (41.95% acc). |
| **XAI (DL)** | Standard Grad-CAM Saliency Maps | **DONE** | [`dl/explainability/gradcam.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/explainability/gradcam.py), [`explainability_service.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/explainability_service.py#L200-L220) | Hooks `layer4[-1]`, extracts activations, averages gradients, computes ReLU CAM, overlays on input image. |
| **XAI (DL)** | Grad-CAM++ | **MISSING** | Codebase wide search | **Zero code implemented.** Only standard Grad-CAM exists in code. UI labels claiming "Grad-CAM++" are cosmetic placeholders. |
| **XAI (DL)** | LIME (Local Interpretable Explanations) | **MISSING** | Codebase wide search | **Zero code implemented.** No LIME dependencies or scripts exist anywhere in the repository. |
| **FL Core** | Local Training Refactor (`train_local`) | **DONE** | [`dl/federated/local_training.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/local_training.py#L32-L135) | Trains ResNet18 locally on hospital DataLoader; computes metrics; returns weights + sample counts. |
| **FL Core** | Dataset Partitioning (IID / Non-IID) | **DONE** | [`dl/federated/partition.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/partition.py#L28-L125) | Partitions dataset into Apollo, Manipal, AIIMS folders. Verified reproducible split. |
| **FL Core** | Flower FL Client & Server | **DONE** | [`dl/federated/client.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/client.py), [`server.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/server.py), [`strategy.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/strategy.py) | `StoneSenseFLClient` and `StoneSenseFedAvg` coordinate distributed parameter exchange without transmitting raw scans. |
| **FL Core** | Model Versioning Checkpoint Manager | **DONE** | [`dl/federated/model_manager.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/model_manager.py) | Checkpoints `resnet18_fed_round_001.pth` through `029.pth` saved and tracked in `stonesense.db`. |
| **FL Core** | Hospital ↔ Global Synchronization | **DONE** | [`dl/federated/strategy.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/strategy.py#L359), [`developer.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/developer.py#L396) | Updates `Hospital.current_model_version = version_tag` on round completion and manual deploy/rollback. |
| **FL Core** | End-to-End Local Simulation | **DONE** | [`dl/federated/simulate.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/simulate.py) | Multi-hospital simulation verified across 3 rounds and 29 live rounds in SQLite database. Real per-client training confirmed. |
| **Governance** | Automated Deployment Gate | **DONE** | [`backend/app/services/deployment_gate.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/deployment_gate.py) | Enforces accuracy $\ge 0.95$, F1 $\ge 0.94$, stone recall $\ge 0.96$, tumor recall $\ge 0.98$, regression $\le 0.02$. Explicitly rejects candidates trained on leaky partitions. |
| **Governance** | Single-Click Deploy & Rollback | **DONE** | [`backend/app/api/v1/routes/developer.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/developer.py#L360-L530) | Atomically switches `is_deployed` flag and logs cryptographic SHA-256 audit entry. |
| **Security** | Authentication & Scoped RBAC | **DONE** | [`backend/app/core/auth.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/core/auth.py), [`security.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/core/security.py) | JWT tokens with HMAC SHA-256; role validation (`hospital_user`, `developer`, `admin`). Scopes queries to user's assigned hospital. |
| **Security** | Cryptographic Audit Logging | **DONE** | [`backend/app/db/models.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/db/models.py#L210), [`auth.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/core/auth.py#L180-L215) | `AuditLog` table stores timestamp, actor user, hospital scope, action, SHA-256 hash. |
| **Security** | PII Encryption / Pseudonymization | **MISSING** | [`backend/app/db/models.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/db/models.py#L80-L100) | Patient names, phones, and emails are currently stored in plain text in the relational store. |
| **Safety** | CT Input Validation & OOD Filter | **PARTIAL** | [`predict_router.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/predict_router.py#L120-L165) | Image decoding occurs, but lacks file size limit, MIME whitelist, and Out-of-Distribution rejection for non-kidney images. |
| **Clinical UI** | Multi-Modal Evidence Concordance | **PARTIAL** | [`HospitalDashboard.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/pages/HospitalDashboard.tsx#L950-L971) | Surfaced on UI, but uses hardcoded static text ("Moderate risk 71%", "Stone detected 94.2%"). Backend multi-modal rule agreement engine is missing. |

---

## 3. Test Suite & Quality Verification Results

### Backend Test Results (Pytest + Coverage)
```text
Command: backend\.venv\Scripts\python.exe -m pytest -v --cov=backend/app
Results: 106 passed, 1 skipped, 148 warnings in 123.32s (0:02:03)
Exit Code: 0
```
- **Total Tests Collected:** 107
- **Skipped Test:** `testing/test_partition_leakage.py` (Skipped intentionally when raw image partition archives are not unpacked locally, as documented in CI configuration).
- **Code Coverage Breakdown:**
  - `backend/app/db/models.py`: 100%
  - `backend/app/schemas/*`: 100%
  - `backend/app/services/deployment_gate.py`: 87%
  - `backend/app/services/federated_coordinator.py`: 84%
  - `backend/app/api/v1/routes/auth.py`: 93%
  - `backend/app/api/v1/routes/developer.py`: 73%
  - `backend/app/api/v1/routes/predict_router.py`: 73%
  - **Overall Backend Application Coverage:** **72%** (1,950 / 2,699 statements).

### Frontend Verification Results
- **TypeScript Type Checking (`npx tsc --noEmit`):** Passed with code `0` (0 errors).
- **ESLint (`npm run lint`):** Passed with code `0` (0 errors, 10 warnings on React hook exhaustive deps).
- **Production Build (`npm run build`):** Passed with code `0`. Generated optimized assets (`dist/assets/index-lMZ72foU.js` [415 kB], `dist/assets/index-CkeiwB3w.css` [41 kB]) in 11.33 seconds.

---

## 4. Open Bugs & Technical Debt (Ranked by Severity)

1. **[CRITICAL] Plaintext Patient PII in Database:**
   - *Impact:* HIPAA/GDPR non-compliance.
   - *Location:* [`backend/app/db/models.py:Patient`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/db/models.py#L83) (`name`, `phone`, `email`).
   - *Remediation:* Implement AES-256 field encryption using SQLAlchemy TypeDecorator or pseudonymize fields before clinical deployment.
2. **[HIGH] Lack of Image Upload Size & Out-of-Distribution (OOD) Checks:**
   - *Impact:* Denial of Service risk via giant file uploads; misclassification of non-medical images (e.g. photos of pets classified as kidney stones).
   - *Location:* [`backend/app/api/v1/routes/predict_router.py:predict_ct_image`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/predict_router.py#L121).
   - *Remediation:* Enforce max file size (e.g. 10 MB), validate image dimensions/MIME type, and implement maximum softmax probability / energy-based OOD detector.
3. **[HIGH] Hardcoded Multi-Modal Agreement in UI:**
   - *Impact:* Misleading clinical evidence display.
   - *Location:* [`frontend/src/pages/HospitalDashboard.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/pages/HospitalDashboard.tsx#L955-L970).
   - *Remediation:* Create a backend `/api/v1/predict/concordance` endpoint calculating deterministic multi-modal agreement rules from the live predictions.
4. **[MEDIUM] Git Repository Bloat (Over 1 GB of Checkpoints Tracked):**
   - *Impact:* Slow clones, repository bloat.
   - *Location:* `dl/models/federated/resnet18_fed_round_001.pth` through `022.pth` committed to git history before `.gitignore` pattern was applied.
   - *Remediation:* Coordinate git history filter (`git-filter-repo` / `BFG Repo-Cleaner`) to purge historical `.pth` blobs and store weights in external storage (S3/GCS/DVC).
5. **[MEDIUM] Inconsistent scikit-learn Serialization Warning:**
   - *Impact:* Non-blocking `InconsistentVersionWarning` emitted when unpickling candidate models across scikit-learn 1.9.0 vs 1.9.1.
   - *Remediation:* Re-save pipelines in virtual environment matching current production dependency lock.
6. **[LOW] Broken Navigation Card Links on Legacy `/home` Page:**
   - *Impact:* User confusion. Clicking cards for `/risk-prediction` or `/stone-detection` redirects to `/` due to missing `:patientId` parameter.
   - *Location:* [`frontend/src/pages/HomePage.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/pages/HomePage.tsx#L19-L27).
   - *Remediation:* Update links to point to `/hospital-dashboard` or provide interactive patient selection.
