# StoneSense-AI Comprehensive Repository Audit Report

**Date of Audit:** October 12, 2026  
**Auditor:** Senior Engineering Code Audit  
**Target Environment:** Windows 11 / Python 3.12 / Node.js 20 / React 18  
**Repository Branch:** `main`  

---

## 1. Executive Summary

1. StoneSense-AI is a functional, operational multi-modal research platform featuring a 52-endpoint FastAPI backend and a React 18 / Tailwind CSS clinical frontend.
2. The active deployed tabular model is Logistic Regression v001 ($ROC\text{-}AUC = 0.8387 \pm 0.0940$, 1-SE rule winner across 50-fold nested CV); ResNet18 CT classifier achieves 86.94% accuracy on leak-free grouped test slices (96.48% scan accuracy).
3. Decentralized federated learning is fully implemented with real per-client training (Flower `NumPyClient` + `StoneSenseFedAvg`), with 29 completed live rounds persisted in `stonesense.db`.
4. Previous marketing claims of 98.5% CT accuracy and 91.67% tabular accuracy were based on discredited leaky slice-shuffled splits and an unstratified 12-sample test subset, respectively.
5. Critical remediation priorities include encrypting plaintext patient PII, adding image upload size/OOD validation, and wiring dynamic server-side evidence concordance.

---

## 2. Feature Status Table

| Feature / PRD Requirement | Status | Evidence File & Line References | Audit Verdict & Real Runtime Behavior |
| :--- | :---: | :--- | :--- |
| **XGBoost Risk Model** | **DONE** | [`ml/training/train_risk_model.py`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/training/train_risk_model.py#L32-L100), [`ml/models/versions/xgboost_v001.pkl`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/models/versions/xgboost_v001.pkl) | Model trained and registered ($ROC\text{-}AUC = 0.8076$). Subordinated to Logistic Regression baseline under 1-SE selection rule. |
| **Active Tabular Model Loading** | **DONE** | [`backend/app/services/model_loader.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/model_loader.py#L153-L175) | Loads active model (`logistic_regression_v001`) and preprocessing pipeline artifact (`preprocessing_pipeline.pkl`) at server startup. |
| **SHAP Explainability** | **DONE** | [`ml/explainability/shap_explainer.py`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/explainability/shap_explainer.py#L79-L90), [`backend/app/services/explainability_service.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/explainability_service.py#L80-L92) | Dynamic explainer switches between `LinearExplainer` and `TreeExplainer`. Top features, attributions, and directions exposed via API and PatientForm UI. |
| **ResNet18 CT Classifier** | **DONE** | [`dl/training/train_resnet18.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/training/train_resnet18.py), [`backend/app/services/model_loader.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/model_loader.py#L120-L146) | 4-class classifier (`Cyst`, `Normal`, `Stone`, `Tumor`). Weights loaded from `dl/models/federated/latest.pth` or `kidney_resnet18.pth`. |
| **Grad-CAM Saliency Maps** | **DONE** | [`dl/explainability/gradcam.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/explainability/gradcam.py), [`backend/app/services/explainability_service.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/explainability_service.py#L200-L220) | Standard Grad-CAM (Selvaraju et al., 2017) hooking `model.layer4[-1]`. Generates overlay image returned via `/static/gradcam/`. |
| **Grad-CAM++** | **MISSING** | Codebase-wide ripgrep search | **Zero code implemented.** Only standard Grad-CAM exists in Python. UI labels were cosmetic placeholders. |
| **LIME** | **MISSING** | Codebase-wide ripgrep search | **Zero code implemented.** No LIME dependencies or scripts exist anywhere in the repository. |
| **Backend REST Routes** | **DONE** | [`backend/app/main.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/main.py#L80-L95) | 52 HTTP endpoints registered across `/auth`, `/hospital`, `/developer`, `/predict`, `/patients`, `/admin`, and `/models`. |
| **Frontend Routing & UI** | **DONE** | [`frontend/src/App.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/App.tsx#L16-L122) | 11 routed views with protected role guards. Landing, Login, Hospital Dashboard, Developer Console, and FL pages fully wired to API. |
| **Rule-Based Evidence Concordance** | **DONE** | [`concordance_service.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/concordance_service.py), [`predict_router.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/predict_router.py#L26-L52), [`ConcordanceCard.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/components/ConcordanceCard.tsx) | Server-side pure deterministic concordance engine evaluating live ML risk and CT imaging streams across 6 closed outcome codes with strict precedence rules and regulatory disclaimer. Hardcoded numbers eliminated from Hospital Dashboard. |
| **CT Input Validation & OOD Gate** | **DONE** | [`image_utils.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/utils/image_utils.py), [`predict_router.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/predict_router.py#L150-L245), [`test_ct_upload_hardening.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/tests/test_ct_upload_hardening.py) | Bounded stream reading (HTTP 413 max 10 MB), content-based magic bytes check (HTTP 415 PNG/JPEG only), decompression bomb protection, degenerate dimension & aspect ratio checks (HTTP 422), and predictive entropy / confidence OOD gate (status='needs_review'). |
| **A1: Local Training Refactor** | **DONE** | [`dl/federated/local_training.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/local_training.py#L32-L135) | `train_local()` function trains model on isolated local DataLoader; returns updated parameters, metrics dict, and sample counts. |
| **A2: Dataset Partitioning** | **DONE** | [`dl/federated/partition.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/partition.py#L28-L125) | Partitions CT dataset into Apollo, Manipal, AIIMS folders. Supports IID and Non-IID distributions with reproducible seeding. |
| **A3: Flower FL Client** | **DONE** | [`dl/federated/client.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/client.py#L51-L180) | `StoneSenseFLClient` implements `get_parameters`, `set_parameters`, `fit`, `evaluate`. Performs true local training; zero raw scan transfer. |
| **A4: Flower Server + FedAvg** | **DONE** | [`dl/federated/server.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/server.py), [`dl/federated/strategy.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/strategy.py) | `StoneSenseFedAvg` strategy aggregates client weight tensors using sample-weighted averaging. |
| **A5: Model Version Manager** | **DONE** | [`dl/federated/model_manager.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/model_manager.py#L22-L85) | Saves immutable versioned checkpoints (`resnet18_fed_round_00N.pth`) and updates `latest.pth`. Checkpoints 001–029 verified on disk. |
| **A6: Database Persistence** | **DONE** | [`backend/app/db/models.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/db/models.py) | Mapped SQLAlchemy models for `Hospital`, `FederatedRound`, `HospitalTrainingRun`, `ModelVersion`, `InferenceLog`, `AuditLog`. |
| **A7: Hospital Model Sync** | **DONE** | [`dl/federated/strategy.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/strategy.py#L359), [`developer.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/api/v1/routes/developer.py#L396) | Updates `Hospital.current_model_version = version_tag` on round completion and manual deploy/rollback. |
| **A8: End-to-End FL Simulation** | **DONE** | [`dl/federated/simulate.py`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/simulate.py#L127-L315) | Multi-hospital simulation runner executes complete rounds, aggregates weights, and decreases loss. 29 completed rounds in SQLite store. |
| **B1–B5: FL APIs, Tables, UI** | **DONE** | `developer.py`, `hospital.py`, `DeveloperDashboard.tsx`, `HospitalFederatedLearning.tsx`, `federatedApi.ts` | Complete REST exposure, analytics logging tables, responsive dashboards, and typed TypeScript service layer. |
| **B6: FL Documentation** | **DONE** | `docs/FEDERATED_LEARNING.md`, `README.md` | System topology, client-server event sequence, and simulation execution documented. |
| **B7: FL Research Comparison** | **PARTIAL** | `partition.py`, `docs/report_figures/fig_8_3_federated_training_convergence.png` | IID and Non-IID partitioning logic implemented; comparative convergence charts generated; formal writeup pending. |
| **Automated Deployment Gate** | **DONE** | [`backend/app/services/deployment_gate.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/deployment_gate.py) | Enforces accuracy $\ge 0.95$, F1 $\ge 0.94$, stone recall $\ge 0.96$, tumor recall $\ge 0.98$, regression $\le 0.02$. Explicitly rejects leaky partitions. |
| **Authentication & RBAC** | **DONE** | [`backend/app/core/auth.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/core/auth.py), [`security.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/core/security.py) | JWT access/refresh tokens; role checks (`hospital_user`, `developer`, `admin`); queries strictly scoped to assigned `hospital_id`. |
| **Patient Record Encryption (PII)**| **MISSING**| [`backend/app/db/models.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/db/models.py#L80-L100) | Patient names, phone numbers, and emails stored in plain text in the relational store. |
| **Infrastructure & CI** | **DONE** | [`docker-compose.yml`](file:///c:/Users/akhil/projects/StoneSense-AI/docker-compose.yml), [`.github/workflows/ci.yml`](file:///c:/Users/akhil/projects/StoneSense-AI/.github/workflows/ci.yml) | Docker Compose orchestration, Alembic migrations, pinned `requirements.txt`, and GitHub Actions CI. |

---

## 3. Commands Executed & Real Outputs

### 3.1 Backend Test Suite Execution
```powershell
Command: .\backend\.venv\Scripts\python.exe -m pytest -v --cov=backend/app
Exit Code: 0
Duration: 123.32s
Output Summary:
==================== 106 passed, 1 skipped, 148 warnings in 123.32s ====================
Total items: 107.
Skipped item: testing/test_partition_leakage.py (Skipped because raw dataset partition archive is gitignored).
Code Coverage: 72% across backend/app (1,950 / 2,699 statements).
Key Module Coverage:
- app/db/models.py: 100%
- app/schemas/*: 100%
- app/services/deployment_gate.py: 87%
- app/services/federated_coordinator.py: 84%
- app/api/v1/routes/auth.py: 93%
- app/api/v1/routes/developer.py: 73%
- app/api/v1/routes/predict_router.py: 73%
```

### 3.2 Frontend TypeScript Type Check
```powershell
Command: cd frontend; npx tsc --noEmit
Exit Code: 0
Output: (Empty - Zero TypeScript type errors detected)
```

### 3.3 Frontend ESLint Execution
```powershell
Command: cd frontend; npm run lint
Exit Code: 0
Output Summary:
0 problems (0 errors, 0 warnings)
(All former 10 react-hooks/exhaustive-deps and react-refresh warnings resolved in Phase 4 hardening).
```

### 3.4 Frontend Production Build
```powershell
Command: cd frontend; npm run build
Exit Code: 0
Duration: 11.33s
Output:
✓ 116 modules transformed.
dist/index.html                   0.61 kB │ gzip:   0.36 kB
dist/assets/index-CkeiwB3w.css   41.85 kB │ gzip:   7.77 kB
dist/assets/index-lMZ72foU.js   415.49 kB │ gzip: 113.51 kB
✓ built in 11.33s
```

### 3.5 Runtime Service Probe (Health & Models Endpoints)
```text
Endpoint: GET /api/v1/health
Status Code: 200 OK
Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "resnet18_ct_classification": true,
    "xgboost_risk_prediction": true,
    "preprocessing_pipeline": true
  },
  "hardware": {
    "cuda_available": false,
    "active_device": "cpu"
  }
}

Endpoint: GET /api/v1/models (Authenticated: admin@stonesense.ai)
Status Code: 200 OK
Response Summary:
- dl_resnet18: Loaded: true, Accuracy: 0.8694, F1 Macro: 0.8251, Stone Recall: 0.6570, Tumor Recall: 1.0000.
  Disclaimer: "Research Prototype Only — Not for Clinical Diagnosis or Treatment Planning."
- dl_models: ResNet18 (READY), YOLO26 (READY), DINOv3 (READY), QKNN (READY).
- ml_tabular: Name: "Urine-Analysis-Risk-Predictor (logistic_regression_v001)", Validation Accuracy: 0.8387, Validation F1: 0.6937, Loaded: true.
```

---

## 4. Bugs and Technical Debt (Ranked by Severity)

1. **[CRITICAL - Security] Plaintext Patient PII in Database:**
   - *Description:* The `patients` table stores `name`, `phone`, and `email` as unencrypted VARCHAR strings.
   - *Impact:* Potential HIPAA / GDPR violation if deployed with real clinical data.
   - *Recommendation:* Implement AES-256 column-level encryption or salted pseudonymization before clinical deployment.
2. **[HIGH - Robustness] Unchecked Image Uploads & Missing OOD Filter:**
   - *Description:* `/api/v1/predict/image` reads arbitrary uploaded files into memory without enforcing file size limits, MIME type verification, or Out-of-Distribution (OOD) checks for non-kidney images.
   - *Impact:* Denial of Service risk via giant file uploads; non-medical images (e.g., photos of pets) are assigned 100% confidence across the 4 classes.
   - *Recommendation:* Add file size middleware (max 10 MB), validate image format headers, and implement an energy-based or confidence-based OOD detector.
3. **[HIGH - Clinical Integrity] Hardcoded Multi-Modal Concordance in UI:**
   - *Description:* The "Trustworthy assessment" step on `HospitalDashboard.tsx` displays static placeholder text ("Moderate risk 71%", "Stone detected 94.2%").
   - *Impact:* Misleads clinicians into believing cross-modal agreement is dynamically validated when it is currently hardcoded text.
   - *Recommendation:* Implement backend `/api/v1/predict/concordance` evaluating actual independent risk and CT inference outputs.
4. **[MEDIUM - Repository Hygiene] Over 1 GB of Binary Checkpoints Committed to Git History:**
   - *Description:* Checkpoint files `resnet18_fed_round_001.pth` through `022.pth` (44.8 MB each) are tracked in git history because they were committed before `.gitignore` added `dl/models/**/*.pth`.
   - *Impact:* Slow clones, bloated repository size.
   - *Recommendation:* Purge historical checkpoints using `git-filter-repo` and store model weights in external object storage (S3/GCS/DVC).
5. **[MEDIUM - Tooling] Inconsistent scikit-learn Serialization Warning:**
   - *Description:* Unpickling models emits `InconsistentVersionWarning: Trying to unpickle estimator LabelBinarizer from version 1.9.1 when using version 1.9.0`.
   - *Recommendation:* Re-save pipelines and align virtual environment dependency versions.
6. **[LOW - Frontend] Broken Card Links on Legacy `/home` Page:**
   - *Description:* Card buttons on `/home` link to `/risk-prediction` and `/stone-detection` without a `:patientId` parameter, causing an immediate redirect to `/`.
   - *Recommendation:* Route cards to `/hospital-dashboard` or provide a patient selection modal.

---

## 5. Stray & Unclear Files Inventory (Awaiting User Approval)

The following files at the repository root and subdirectories were analyzed:

| File / Folder Path | What It Is | Active Code References | Recommendation | Action Status |
| :--- | :--- | :--- | :--- | :---: |
| `AppLayout.tsx` (Repo Root) | Interim drop-in layout file with demo toggle | None. File contains header comment: `// Drop at: frontend/src/components/layout/AppLayout.tsx`. File was copied and styled into `frontend/src/components/layout/AppLayout.tsx`. | **Delete** (superseded by `frontend/src/components/layout/AppLayout.tsx`). | Awaiting Approval |
| `Login.tsx` (Repo Root) | Early mock login component with note: `// NOTE: this has no real authentication yet` | None. Replaced by production JWT `LoginPage.tsx` in `frontend/src/pages/`. | **Delete** (superseded by `frontend/src/pages/LoginPage.tsx`). | Awaiting Approval |
| `antigravity_fix_prompt.md` | One-off instruction prompt for integrating layout and theme | None. Historical task prompt. | **Move** to `docs/archive/` or **Delete**. | Awaiting Approval |
| `frontend_repair_report.md` | Historical repair report detailing Vite/Tailwind fixes | None. Completed task report. | **Move** to `docs/archive/` or **Keep** as audit artifact. | Awaiting Approval |
| `Hospital overview patients mockup.html` | Static HTML mockup of patient overview | None. Identical to `hospital_overview_patients_mockup.html` except for 1 commented line. | **Delete** duplicate file. | Awaiting Approval |
| `hospital_overview_patients_mockup.html` | Static HTML mockup of patient overview | None. Reference wireframe. | **Move** to `docs/archive/` or **Keep** as visual design reference. | Awaiting Approval |
| `dashboards_preview.html` (Repo Root) | Static HTML preview of both dashboards | None. Identical byte-for-byte SHA duplicate of `stonesense-dashboard-overhaul/.../dashboards_preview.html`. | **Delete** root duplicate. | Awaiting Approval |
| `stonesense-dashboard-overhaul/` | Standalone bundle containing backend and frontend overhaul code | All code inside this bundle was integrated into `backend/` and `frontend/` in commit `54c1ec9`. | **Delete** or **Archive** outside working tree. | Awaiting Approval |
| `stonesense.db` (Repo Root) | SQLite runtime database containing 13 tables (hospitals, users, rounds, logs) | Referenced by `DEFAULT_SQLITE_PATH` in `backend/app/db/database.py` and `alembic.ini`. Ignored in `.gitignore`. | **Keep on disk, NEVER commit to git.** | Operational Store |

---

## 6. Ranked List of Next 10 Engineering Tasks

1. **[Security] Implement Column-Level AES-256 Encryption for Patient PII**: Encrypt `name`, `phone`, and `email` fields in `Patient` table using SQLAlchemy TypeDecorator and environment-driven master key.
2. **[Clinical Safety - RESOLVED] Dynamic Server-Side Rule Concordance Engine**: Implemented in [`backend/app/services/concordance_service.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/services/concordance_service.py) and exposed via `POST /api/v1/predict/concordance`. Deterministic evaluation across independent streams with 6 closed machine codes, explicit precedence rules, and unpaired-dataset disclaimer. Fully connected to Hospital Dashboard via [`ConcordanceCard.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/components/ConcordanceCard.tsx); all static placeholder numbers removed.
3. **[Robustness - RESOLVED] Input Hardening for CT Uploads**: Implemented in [`backend/app/utils/image_utils.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/utils/image_utils.py) and `/api/v1/predict/image`. Bounded 10 MB payload limits (HTTP 413), magic bytes validation (PNG/JPEG only, HTTP 415), decompression bomb protection (HTTP 413), degenerate dimension / aspect ratio checks (HTTP 422), and predictive entropy / confidence OOD heuristic gate (status='needs_review'). 12/12 validation tests passing.
4. **[Repository Hygiene] Purge Historical Model Blobs from Git**: Use `git-filter-repo` to strip 22 historical `.pth` files from git commit history and store model checkpoints in cloud storage or DVC.
5. **[ML Dependency Lock] Re-serialize Scikit-Learn Pipelines**: Align scikit-learn version across all environments (1.9.0 vs 1.9.1) and re-serialize `preprocessing_pipeline.pkl` and `scaler.pkl` to eliminate `InconsistentVersionWarning`.
6. **[Frontend Polish] Fix Legacy Navigation Links**: Update `/home` card buttons to route cleanly to `/hospital-dashboard` with active patient context rather than triggering `/404` redirects.
7. **[Frontend Polish] Clean React Hook Dependency Warnings**: Address 10 `exhaustive-deps` warnings in `HospitalDashboard.tsx`, `HospitalFederatedLearning.tsx`, and `DeveloperDashboard.tsx`.
8. **[Federated Scalability] Transition Federated Coordinator to Celery / Redis Broker**: Move `FederatedCoordinator` off an in-process thread to a robust distributed task queue supporting multi-worker deployments and graceful restarts.
9. **[Governance] Implement Automated Candidate Gate Evaluation CLI**: Create a developer CLI command (`evaluate-candidate.py`) triggering full deployment gate validation against held-out partitions with formatted pass/fail markdown reports.
10. **[Research / QKNN] Explore Higher-Qubit Quantum Feature Map**: If quantum research is resumed, replace the 4-qubit PCA bottleneck with a multi-qubit angle-embedding ansatz evaluated on a larger clinical subset.

---

## 7. Unverified Claims & Verification Plan

| Claim / Metric | Current Audit Status | Why It Could Not Be Verified | What Is Needed to Verify It |
| :--- | :---: | :--- | :--- |
| **"91.67% Tabular Accuracy"** | **UNVERIFIED (Discredited)** | No backing evaluation artifact on disk supports 91.67% across full data. The metric originated from a legacy 12-sample test subset ($11/12 = 91.67\%$). 50-fold nested repeated CV verified real mean accuracy as 74.67% (Logistic Regression) and 77.14% (XGBoost). | Discredited. Metric has been removed from all documentation in favor of 50-fold nested CV results. |
| **"98.5%–99.63% CT Accuracy"** | **UNVERIFIED (Discredited)** | Forensic hash audit discovered that this historical score was measured on a random slice shuffle with 227 duplicate pairs and 703 leaking perceptual clusters crossing train/test splits. | Discredited. Leak-free grouped test evaluation established true slice accuracy as 86.94% (96.48% scan accuracy). Removed from documentation. |
| **"Grad-CAM++"** | **UNVERIFIED (Absent)** | No higher-order derivative calculations or Grad-CAM++ classes exist in the codebase. Only standard Grad-CAM exists. | Implement `pytorch_grad_cam.GradCAMPlusPlus` in `dl/explainability/gradcam.py` or update all labels to "Grad-CAM". (Labels have been corrected in docs). |
| **"LIME Explainability"** | **UNVERIFIED (Absent)** | No LIME library, scripts, or outputs exist anywhere in the repository. | Remove all references to LIME from marketing text and repo descriptions. (Completed in docs overhaul). |
| **True Patient-Level CT Separation** | **UNVERIFIED** | The public axial CT dataset does not provide DICOM patient identifiers or acquisition serials. While 64-bit DCT pHash clustering ($pHash \le 2$) eliminates near-duplicate slice leakage, true patient separation cannot be mathematically proven without source metadata. | Acquire raw DICOM scans with anonymized Patient ID tags or prospectively collected clinical scans with verifiable patient identifiers. |

---

## 8. Summary of Files Created and Modified

### Modified Files:
1. [`README.md`](file:///c:/Users/akhil/projects/StoneSense-AI/README.md): Complete overhaul to match verified code reality, honest metrics, registered endpoints, and prominent disclaimers.
2. [`DEVELOPER_GUIDE.md`](file:///c:/Users/akhil/projects/StoneSense-AI/DEVELOPER_GUIDE.md): Corrected stale paths (removed references to `backend/main.py`), updated uvicorn execution commands, test steps, and router registration instructions.
3. [`prd.md`](file:///c:/Users/akhil/projects/StoneSense-AI/prd.md): Appended verified implementation status table for items A1–A8 and B1–B7 and version changelog without altering original requirements.
4. [`frontend/src/services/developerApi.ts`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/services/developerApi.ts): Fixed trivial TypeScript compilation error in `fallbackModelVersions` (restored `status: 'archived', environment: 'staging'`).
5. [`frontend/README.md`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/README.md): Overhauled generic Vite template boilerplate with real frontend architecture, components, and commands.

### Newly Created Files:
1. [`docs/STATUS.md`](file:///c:/Users/akhil/projects/StoneSense-AI/docs/STATUS.md): Complete feature verification matrix with file/line evidence links, test execution results, open bugs, and technical debt.
2. [`docs/ARCHITECTURE.md`](file:///c:/Users/akhil/projects/StoneSense-AI/docs/ARCHITECTURE.md): Accurate end-to-end Mermaid architecture diagram, subsystem breakdowns, and database schema mappings.
3. [`MODEL_CARD.md`](file:///c:/Users/akhil/projects/StoneSense-AI/MODEL_CARD.md): Dedicated model cards for Tabular Risk models, ResNet18 CT, and supplementary architectures with verified metrics and clinical non-intended use disclaimers.
4. [`backend/README.md`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/README.md): Subsystem documentation for FastAPI backend, routes, database, and test execution.
5. [`ml/README.md`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/README.md): Subsystem documentation for ML pipeline, 50-fold nested CV, dynamic SHAP, and model registry.
6. [`dl/README.md`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/README.md): Subsystem documentation for CT classification, grouped leak-free benchmarks, Grad-CAM, and Flower federated learning.
7. [`infra/README.md`](file:///c:/Users/akhil/projects/StoneSense-AI/infra/README.md): Infrastructure documentation for Docker Compose, Dockerfiles, Nginx, and environment variables.
8. [`testing/README.md`](file:///c:/Users/akhil/projects/StoneSense-AI/testing/README.md): Test suite map, execution commands, and verified audit results.
9. [`CHANGELOG.md`](file:///c:/Users/akhil/projects/StoneSense-AI/CHANGELOG.md): Comprehensive project changelog based on git commit history.
10. [`docs/AUDIT_REPORT.md`](file:///c:/Users/akhil/projects/StoneSense-AI/docs/AUDIT_REPORT.md): This comprehensive final audit report.
