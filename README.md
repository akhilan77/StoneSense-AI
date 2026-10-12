# StoneSense-AI

[![StoneSense-AI CI](https://github.com/akhilan77/StoneSense-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/akhilan77/StoneSense-AI/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.12+-FF9800.svg)](https://flower.ai)

> [!CAUTION]
> ### ⚠️ RESEARCH PROTOTYPE DISCLAIMER — NOT FOR CLINICAL USE
> **StoneSense-AI is an experimental academic research prototype and proof-of-concept software system.**  
> It has **NOT** undergone prospective clinical validation, clinical trials, or regulatory certification (FDA 510(k), CE-IVD, CDSCO, or HIPAA/GDPR clinical clearance).  
> **IT MUST NOT BE USED FOR PRIMARY CLINICAL DIAGNOSIS, PATIENT TRIAGE, SURGICAL PLANNING, OR CLINICAL TREATMENT DECISIONS.**

> [!IMPORTANT]
> ### 📊 CRITICAL DATASET LIMITATIONS & GENERALIZATION NOTICE
> 1. **Urine Biochemistry (Tabular) Cohort**: Trained on a historical dataset of **79 unique patient observations** (cleaned from the 80-row Andrews & Lyne bench test cohort). Due to this small sample size, metrics carry broad confidence intervals ($95\%\text{ CI: }[0.7327, 0.9176]$ for Logistic Regression ROC-AUC).
> 2. **CT Scan Slices**: Derived from a public Kaggle dataset containing 12,446 axial CT slice images across 4 classes (`Cyst`, `Normal`, `Stone`, `Tumor`). The original dataset **does not provide patient or volume identifiers**. To prevent near-duplicate slice leakage, splits were grouped using perceptual hashing ($pHash \le 2$ Hamming distance connected components, yielding 979 cluster units). True patient-level isolation across different hospital systems remains **unverified** without raw DICOM metadata.

---

## Executive Overview

StoneSense-AI is a multi-modal Explainable AI (XAI) decision-support platform designed to assist research urologists in evaluating kidney stone risk and axial CT imaging. It provides two independent, decoupled evidence streams:
1. **Biomarker Risk Analysis**: Evaluates 6 routine urine biochemistry parameters (`gravity`, `ph`, `osmo`, `cond`, `urea`, `calc`) using statistical risk models (Logistic Regression baseline, Random Forest, XGBoost) explained via **SHAP** feature attributions.
2. **Axial CT Kidney Slices**: Classifies CT slices using transfer-learning convolutional networks (**ResNet18**), visualized via **Grad-CAM** saliency overlays.
3. **Decentralized Federated Learning**: Simulates multi-hospital collaborative training using the **Flower (flwr)** framework, synchronizing model weights without centralizing raw hospital scan files.

```mermaid
flowchart TD
    Client["React 18 + Vite + Tailwind Frontend\n(Port 5173 / Port 80)"]
    API["FastAPI REST & WebSocket Backend\n(Port 8000)"]
    DB[("Relational SQLite / PostgreSQL Store\n(stonesense.db)")]

    subgraph Presentation["Presentation & Gateway Layer"]
        Client -->|REST API / JWT Auth| API
        Client <-->|WebSocket Real-Time Telemetry| API
        API -->|SQLAlchemy ORM| DB
    end

    subgraph MLStream["Independent Stream 1: Urine Biomarkers"]
        API -->|Predict Urine Risk| RegML["ModelRegistry (Logistic Regression v001)"]
        RegML -->|Feature Attributions| SHAPExplainer["Dynamic SHAP (LinearExplainer / TreeExplainer)"]
    end

    subgraph DLStream["Independent Stream 2: CT Kidney Scans"]
        API -->|Classify CT Scan| ResNet["ResNet18 Classifier (Layer 4)"]
        ResNet -->|Saliency Overlay| GradCAMExplainer["Standard Grad-CAM (Selvaraju et al.)"]
    end

    subgraph FedStream["Federated Learning & Governance"]
        API <-->|Coordinate Rounds| Coord["FederatedCoordinator (Flower FedAvg)"]
        Coord -->|Version Checkpoints| Chk["dl/models/federated/ (latest.pth)"]
        API -->|Deploy / Rollback| Gate["DeploymentGate Service"]
    end
```

---

## Verified Project Status

| Component | Status | Verified Reality & Code Evidence |
| :--- | :---: | :--- |
| **Tabular Risk Pipeline** | **DONE** | 50-fold nested repeated stratified CV implemented in [`ml/training/train_risk_model.py`](ml/training/train_risk_model.py). Model registry in [`ml/risk_models/registry.py`](ml/risk_models/registry.py) active with `logistic_regression_v001` (1-SE winner) and `xgboost_v001`. |
| **SHAP Explainability** | **DONE** | Dynamic explainer (`LinearExplainer` for linear models, `TreeExplainer` for tree models) implemented in [`ml/explainability/shap_explainer.py`](ml/explainability/shap_explainer.py) and [`backend/app/services/explainability_service.py`](backend/app/services/explainability_service.py). Top features and directions returned via API. |
| **CT ResNet18 Classifier** | **DONE** | PyTorch ResNet18 transfer learning pipeline in [`dl/training/train_resnet18.py`](dl/training/train_resnet18.py); audited grouped benchmark in [`dl/training/train_resnet18_audit.py`](dl/training/train_resnet18_audit.py). Weights loaded in backend [`backend/app/services/model_loader.py`](backend/app/services/model_loader.py). |
| **Grad-CAM Saliency Maps** | **DONE** | Standard Grad-CAM (Selvaraju et al., 2017) targeting `layer4[-1]` implemented in [`dl/explainability/gradcam.py`](dl/explainability/gradcam.py), [`backend/app/services/explainability_service.py`](backend/app/services/explainability_service.py), and [`dl/models/wrappers/resnet18_wrapper.py`](dl/models/wrappers/resnet18_wrapper.py). |
| **Grad-CAM++** | **MISSING** | **Not implemented.** Only standard Grad-CAM exists in code; UI labels ("Grad-CAM++") and former diagram notes were cosmetic placeholders. |
| **LIME** | **MISSING** | **Not implemented.** No LIME library, explainer script, or dependencies exist in the repository. |
| **Multi-Model DL Wrappers** | **PARTIAL** | Wrappers implemented in [`dl/models/wrappers/`](dl/models/wrappers/) for YOLO26 (`best.pt`), DINOv3 (feature backbone), and QKNN (quantum simulator). YOLO26 and DINOv3 run inference; QKNN is an experimental low-accuracy prototype (41.95% acc). |
| **Federated Learning (Core A1–A8)** | **DONE** | Local training refactor (`train_local`), dataset partitioning (IID/non-IID), Flower `NumPyClient`, Flower server + `StoneSenseFedAvg`, versioned checkpoints (`resnet18_fed_round_001.pth`–`029.pth`), DB telemetry persistence, and multi-hospital simulation runner in [`dl/federated/`](dl/federated/). Real per-client training confirmed. |
| **Federated UI & APIs (B1–B5)** | **DONE** | 52 REST endpoints registered in FastAPI; Hospital FL UI ([`HospitalFederatedLearning.tsx`](frontend/src/pages/HospitalFederatedLearning.tsx)) and Developer FL Console ([`DeveloperDashboard.tsx`](frontend/src/pages/DeveloperDashboard.tsx)) connected via [`federatedApi.ts`](frontend/src/services/federatedApi.ts) and WebSockets. |
| **Rule-Based Evidence Agreement** | **PARTIAL** | Surfaced on Hospital Dashboard ("Trustworthy assessment" step), but **hardcoded in frontend** with fixed strings ("Moderate risk 71%", "Stone detected 94.2%"). Backend multi-modal concordance calculation is not yet wired. |
| **Image Upload Validation** | **PARTIAL** | Backend accepts image byte payloads and processes them through PIL/Torchvision, but **lacks file size limits, MIME type validation, and Out-of-Distribution (OOD) / non-kidney rejection**. |
| **Authentication & RBAC** | **DONE** | JWT access/refresh tokens with bcrypt password hashing; role-based access control (`hospital_user`, `developer`, `admin`) and hospital scoping enforced in [`backend/app/core/auth.py`](backend/app/core/auth.py). |
| **Patient Record Encryption (PII)** | **MISSING** | Patient names, phone numbers, and emails are currently stored in plain text in the SQLite/PostgreSQL `patients` table. Encryption-at-rest or pseudonymization is not yet implemented. |
| **Infrastructure & CI** | **DONE** | Pinned dependencies in [`requirements.txt`](requirements.txt), Docker Compose ([`docker-compose.yml`](docker-compose.yml)), Alembic migrations, and GitHub Actions CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) with fast Pytest run. |

---

## Verified Evaluation Metrics

All metrics reported below are backed by verified evaluation files committed on disk. Historical unverified or leaky metrics are explicitly flagged.

### 1. Tabular Risk Models (50-Fold Repeated Stratified Nested CV)
*Backing file:* [`ml/models/comparison_results.csv`](ml/models/comparison_results.csv) & [`ml/models/registry_manifest.json`](ml/models/registry_manifest.json)  
*Validation Protocol:* 5 folds $\times$ 10 repeats (50 total cross-validation folds) with row-level bootstrap (1,000 iterations) and 1,000-iteration permutation significance testing.

| Model | Complexity Rank | Mean ROC-AUC (95% CI) | Mean Macro F1 (95% CI) | Mean Accuracy (95% CI) | Mean MCC | Permutation $p$-value | Deployed Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **1 (Simplest)** | **0.8387** `[0.7327, 0.9176]` | **0.6937** `[0.5333, 0.7937]` | **74.67%** `[0.6329, 0.8231]` | **0.4926** | **0.001** | **ACTIVE (1-SE Winner)** |
| **Random Forest** | 2 | 0.8175 `[0.7122, 0.9191]` | 0.7067 `[0.5778, 0.8302]` | 75.62% `[0.6835, 0.8608]` | 0.5152 | 0.001 | Candidate |
| **XGBoost** | 3 | 0.8076 `[0.7080, 0.9156]` | 0.7169 `[0.5862, 0.8308]` | 77.14% `[0.6835, 0.8608]` | 0.5505 | 0.001 | Registered v001 |

> [!NOTE]
> **Why Logistic Regression is Deployed:** Under the **1-Standard-Error rule** with Nadeau-Bengio corrected variance, Logistic Regression achieves the highest mean ROC-AUC (0.8387) and is the parsimonious model chosen by the automated selection pipeline.  
> *Legacy Note on "91.67% Tabular Accuracy":* That historical number was evaluated on a non-stratified, 12-sample test subset ($11/12 = 91.67\%$). It is **unverified** across the broader distribution and is superseded by the 50-fold nested CV result.

---

### 2. Axial CT Kidney Classification Models (Held-Out Test Set)
*Backing files:* [`dl/models/ct/resnet18/metrics.json`](dl/models/ct/resnet18/metrics.json), [`docs/RESNET18_AUDIT_REPORT.md`](docs/RESNET18_AUDIT_REPORT.md), and [`dl/training/train_resnet18_audit.py`](dl/training/train_resnet18_audit.py)  
*Validation Protocol:* 12,446 total images clustered using 64-bit DCT perceptual hashing ($pHash \le 2$). 1,869 held-out test slices (142 cluster units) evaluated strictly once.

| Model Architecture | Evaluation Split Protocol | Slice Accuracy | Slice Macro F1 | Stone Sensitivity (Recall) | Tumor Sensitivity (Recall) | Cluster/Scan Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet18 (Audited Baseline)** | **Grouped $pHash \le 2$ (Zero Duplicate Leakage)** | **86.94%** | **82.51%** | **65.70%** (136/207) | **100.0%** (343/343) | **96.48%** |
| *ResNet18 (Historical Baseline)* | *Slice-level random shuffle (LEAKY)* | *99.63%* | *99.47%* | *97.12%* | *99.71%* | *N/A (Discredited)* |
| **YOLO26 (Supplementary)** | Grouped test partition | 92.19% | 90.47% | 94.20% | 99.13% | — |
| **DINOv3 (Supplementary)** | Grouped test partition | 87.43% | 82.81% | 61.84% | 89.80% | — |
| **QKNN (Experimental Prototype)** | Grouped test partition | 41.95% | 35.89% | 25.60% | 27.70% | — |

> [!WARNING]
> **Discredited 98.5%–99.6% CT Accuracy:** Forensic hash analysis revealed that the historical 99.63% accuracy claim suffered from 227 exact cross-split duplicate pairs and 703 perceptual clusters spanning train and test splits. On true leakage-free held-out test slices, ResNet18 achieves **86.94% slice accuracy**, though majority voting across clustered slices achieves **96.48% scan accuracy**.

---

### 3. Federated Learning Multi-Hospital Convergence
*Backing file:* `stonesense.db` (`federated_rounds` & `hospital_training_runs` tables) and [`docs/report_figures/scripts/gen_chap8_figures.py`](docs/report_figures/scripts/gen_chap8_figures.py)  
*Setup:* 3 simulated hospitals (Apollo, Manipal, AIIMS) running FedAvg aggregation across 29 live completed rounds.

- **Round 1 Global Validation Accuracy**: **97.75%** (Loss: 0.0894)
- **Round 2 Global Validation Accuracy**: **98.66%** (Loss: 0.0464)
- **Round 3 Global Validation Accuracy**: **98.98%** (Loss: 0.0383)
- **Rounds 4–29 Convergence**: Sustained validation accuracy plateau between **98.5% and 99.2%** across hospital validation partitions.

---

## Registered Frontend Pages & Navigation Routes

The frontend is built with React 18, Vite, TypeScript, and Tailwind CSS. The active routing structure in [`frontend/src/App.tsx`](frontend/src/App.tsx) consists of:

| URL Route | Page Component | Role Access | Primary APIs Called | Notes / Flags |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `LandingPage.tsx` | Public | None | Marketing overview, entry point to `/login`. |
| `/login` | `LoginPage.tsx` | Public | `POST /api/v1/auth/login` | Email/password sign-in with quick-fill demo buttons (`dr.aisha`, `dev`, `admin`). |
| `/privacy` | `PrivacyPolicy.tsx` | Public | None | Clinical disclaimer and research privacy terms. |
| `/hospital-dashboard` | `HospitalDashboard.tsx` | `hospital_user`, `developer`, `admin` | `/api/v1/hospital/list`, `/api/v1/patients`, `/api/v1/predict/risk`, `/api/v1/predict/image`, `/api/v1/hospital/{id}/history` | Stepped clinical workflow (biomarkers $\to$ CT scan $\to$ XAI $\to$ assessment) + patient queue. *Note: "Trustworthy assessment" concordance is currently hardcoded text.* |
| `/hospital/federated-learning` | `HospitalFederatedLearning.tsx` | `hospital_user`, `developer`, `admin` | `/api/v1/hospital/{id}/federated-status`, `/api/v1/hospital/{id}/train-local`, WebSocket `/hospital/{id}/federated/ws` | Hospital-side federated client dashboard: local dataset size, training trigger, round status. |
| `/hospital/private-dataset` | `HospitalPrivateDataset.tsx` | `hospital_user`, `developer`, `admin` | `/api/v1/hospital/{id}/dataset-status`, `/api/v1/hospital/{id}/dataset-validate`, `/api/v1/hospital/{id}/dataset-upload` | Upload and validate private hospital CT partitions. |
| `/risk-prediction/:patientId` | `RiskPredictionPage.tsx` | `hospital_user`, `developer`, `admin` | `POST /api/v1/predict/risk`, `GET /api/v1/patients/{id}` | Direct risk prediction form for a specific patient. |
| `/stone-detection/:patientId` | `StoneDetectionPage.tsx` | `hospital_user`, `developer`, `admin` | `POST /api/v1/predict/image`, `GET /api/v1/patients/{id}` | CT image upload and Grad-CAM visualization for a specific patient. |
| `/developer-dashboard` | `DeveloperDashboard.tsx` | `developer`, `admin` | `/api/v1/developer/*` (stats, model-performance, model-versions, deploy, rollback, federated-overview, system-monitoring, audit-logs) | Full developer console with tabs: Overview, Federated Learning, Model Registry & Gate, Access/Audit Logs, System Health. |
| `/home` | `HomePage.tsx` | Public | `GET /api/v1/health`, `GET /api/v1/models` | *Legacy marketing view.* Card links to `/risk-prediction` and `/stone-detection` lack `:patientId` parameter and redirect to `/`. |
| `/404` | `NotFoundPage.tsx` | Public | None | Fallback route. |

---

## Backend REST API Routes Summary

The backend exposes **52 endpoints** organized into modular FastAPI routers registered in [`backend/app/main.py`](backend/app/main.py):

- **Health & Info**: `GET /api/v1/health` (public), `GET /api/v1/models` (requires auth), `GET /api/v1/models/dl`.
- **Authentication**: `POST /api/v1/auth/login`, `POST /api/v1/auth/refresh`, `GET /api/v1/auth/me`.
- **Clinical Inference**: `POST /api/v1/predict/risk` (urine biomarkers + SHAP), `POST /api/v1/predict/image` (CT scan + Grad-CAM).
- **Patient Management**: `GET /api/v1/patients`, `POST /api/v1/patients`, `GET /api/v1/patients/{id}`.
- **Hospital Endpoints** (registered at both `/api/v1/hospital` and `/api/v1/hospitals`): `/list`, `/{id}/detail`, `/{id}/current-model`, `/{id}/dataset-status`, `/{id}/dataset-upload`, `/{id}/dataset-validate`, `/{id}/train-local`, `/{id}/training-history`, `/{id}/federated-status`, `/{id}/history`.
- **Developer Endpoints**: `/system-stats`, `/statistics`, `/model-performance`, `/model-versions`, `/model-versions/deploy`, `/model-versions/rollback`, `/federated-overview`, `/federated/rounds/start`, `/federated/rounds/current/status`, `/hospital-participation`, `/round-history`, `/system-monitoring`, `/drift-analysis`, `/system-logs`, `/hospital-logs`.
- **Admin**: `GET /api/v1/admin/audit-logs`.
- **WebSockets**: `/api/v1/ws/federated`, `/api/v1/developer/federated/ws`, `/api/v1/hospital/{id}/federated/ws`.

---

## Local Development & Setup

### Prerequisites
- **Python**: 3.11 or 3.12 (Virtual environments supported in `backend/.venv`, `ml/.venv`, `dl/.venv`)
- **Node.js**: 18.0+ / 20.0+ with npm 9+
- **Git**

### 1. Backend Setup & Startup
The active backend application entrypoint is **`backend/app/main.py`** exposing `app = create_app()`.  
*(Note: An obsolete root-level `backend/main.py` stub was deleted in commit `5a64a58`.)*

```bash
# Windows launcher (recommended):
.\run_backend.bat

# Or manual execution:
cd backend
.\.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Setup & Startup
```bash
# Windows launcher:
.\run_frontend.bat

# Or manual execution:
cd frontend
npm install
npm run dev
```
Frontend dev server runs at `http://localhost:5173`.

### 3. Running Backend Tests & Code Coverage
```bash
# Activate backend virtual environment from repository root:
.\backend\.venv\Scripts\python.exe -m pytest -v --cov=backend/app
```
*Current test suite: 106 passed, 1 skipped (partition leakage check requires local dataset archives), 0 failed (72% code coverage across backend/app).*

### 4. Running Frontend Type Checks & Production Build
```bash
cd frontend
npx tsc --noEmit
npm run lint
npm run build
```

---

## Known Limitations & Technical Debt

1. **Patient Identifier Encryption (PII)**: Patient records in `stonesense.db` store names, phone numbers, and emails in cleartext. Needs column-level encryption or salted pseudonymization before clinical pilot.
2. **Hardcoded Concordance in UI**: The "Trustworthy assessment" step on `HospitalDashboard.tsx` uses static placeholder strings rather than computing dynamic rule agreement from actual prediction outputs.
3. **CT Image Upload Validation**: The `/predict/image` endpoint does not enforce maximum file size limits, MIME type verification, or an Out-of-Distribution (OOD) rejection filter for non-kidney images.
4. **Tracked Model Weights in Git History**: Over 1 GB of binary PyTorch checkpoints (`resnet18_fed_round_001.pth`–`022.pth`) were committed to git history prior to `.gitignore` exclusions.
5. **Single-Process Federated Coordinator**: Live federated rounds triggered via REST use an in-process thread with `threading.Lock`. Production multi-worker setups require a background task broker (e.g. Celery / Redis).

---

## Project Roadmap

- [ ] **Phase 1 (ML & Tabular Hardening)**: Complete — 50-fold nested repeated CV implemented, Logistic Regression v001 registered as deployed 1-SE baseline.
- [ ] **Phase 2 (Dynamic Explainability)**: Complete — Dynamic `LinearExplainer` / `TreeExplainer` operational; standard Grad-CAM operational.
- [ ] **Phase 3 (Deployment Gate & Governance)**: Complete — Automated multi-metric gate rejects unverified and leaky models.
- [ ] **Phase 4 (Security & HIPAA Prep)**: 
  - [x] JWT authentication and role-based access control.
  - [ ] Implement AES-256 field encryption for patient PII (`name`, `phone`, `email`).
  - [ ] Add dynamic server-side rule agreement engine for multi-modal concordance.
- [ ] **Phase 5 (Input Hardening & Safety)**: Add file size limits, image MIME validation, and confidence-based OOD rejection on CT uploads.
- [ ] **Phase 6 (Repository Hygiene)**: Prune historical `.pth` binary blobs from git history using `git-filter-repo` and move to an external artifact store.

---

## License

This project is licensed under the terms of the MIT License (see [LICENSE](LICENSE) file). Pretrained weights and backbones remain subject to their original upstream licenses (Torchvision BSD-3, Meta DINOv3 Research License).
