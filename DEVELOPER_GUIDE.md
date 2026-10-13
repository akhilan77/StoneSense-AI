# StoneSense-AI Developer Guide

Welcome to the **StoneSense-AI Developer Guide**. This document provides verified, technical instructions for environment configuration, daily development workflows, test execution, model retraining, API expansion, and troubleshooting based on the active codebase.

---

## Table of Contents
1. [Architecture & Technology Stack](#architecture--technology-stack)
2. [Environment Setup & Installation](#environment-setup--installation)
3. [Running the Application](#running-the-application)
4. [Database Management & Migrations](#database-management--migrations)
5. [Running Test Suites & Linters](#running-test-suites--linters)
6. [How to Retrain ML Models (Tabular Risk)](#how-to-retrain-ml-models-tabular-risk)
7. [How to Retrain DL Models (CT ResNet18)](#how-to-retrain-dl-models-ct-resnet18)
8. [How to Run Federated Learning Simulations](#how-to-run-federated-learning-simulations)
9. [How to Add New Backend APIs](#how-to-add-new-backend-apis)
10. [How to Add New Frontend Pages & Routes](#how-to-add-new-frontend-pages--routes)
11. [Troubleshooting & Common Issues](#troubleshooting--common-issues)

---

## Architecture & Technology Stack

- **Backend**: Python 3.12, FastAPI 0.115+, SQLAlchemy 2.0 ORM, Uvicorn ASGI server.
  - *Active Entrypoint*: [`backend/app/main.py`](backend/app/main.py) exposing factory `create_app()`.
  - *Database*: SQLite by default (`stonesense.db` at repo root), PostgreSQL supported via `STONESENSE_DATABASE_URL`.
- **Frontend**: Node.js 20+, React 18.3, TypeScript 5.7, Vite 5.4, Tailwind CSS 3.4.
  - *Active Entrypoint*: [`frontend/src/main.tsx`](frontend/src/main.tsx) mounting [`frontend/src/App.tsx`](frontend/src/App.tsx).
- **Machine Learning (Tabular)**: Scikit-learn, XGBoost, SHAP (`ml/`).
- **Deep Learning (CT Slices)**: PyTorch 2.2+, Torchvision, Flower (`flwr`) federated learning (`dl/`).

---

## Environment Setup & Installation

### Windows Automated Setup
The project supports separate Python virtual environments for different components or a unified root environment:

```cmd
:: Automated one-click installation script:
install_all.bat
```

### Manual Virtual Environment Setup (Windows PowerShell)

```powershell
# 1. Setup Backend Environment
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Setup Frontend Dependencies
cd frontend
npm install
cd ..
```

---

## Running the Application

### Option A: Interactive Launcher (Windows)
```powershell
# Interactive console menu with color logging & process management
.\developer_launcher.ps1
```

### Option B: Batch Launchers
- Launch both Backend & Frontend simultaneously: `run_all.bat`
- Launch Backend server only: `run_backend.bat`
- Launch Frontend server only: `run_frontend.bat`
- Terminate all running project processes safely: `stop_all.bat`

### Option C: Manual CLI Commands

```powershell
# Terminal 1: Start FastAPI Backend
cd backend
..\backend\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start React Vite Frontend
cd frontend
npm run dev
```

- **Web Dashboard**: `http://localhost:5173`
- **FastAPI REST API**: `http://127.0.0.1:8000`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs` (active in development)
- **Redoc Documentation**: `http://127.0.0.1:8000/redoc`

---

## Database Management & Migrations

The relational database (`stonesense.db` or PostgreSQL) manages multi-hospital tenants, patient cases, model registry manifests, audit logs, and federated round telemetry.

```powershell
# Apply Alembic schema migrations to latest revision:
.\backend\.venv\Scripts\alembic.exe upgrade head

# Mark existing database as head without reapplying schema:
.\backend\.venv\Scripts\alembic.exe stamp head

# Seed initial demonstration hospitals and user accounts:
.\backend\.venv\Scripts\python.exe -m backend.app.db.seed
```

Demo accounts created by seed script:
- **Hospital User**: `dr.aisha@apex-health.org` / `HospitalPass123!` (Scoped to Apex Health / HOSP-001)
- **Developer**: `dev@stonesense.ai` / `DevPass123!`
- **Administrator**: `admin@stonesense.ai` / `AdminPass123!`

---

## Running Test Suites & Linters

### Backend Tests & Code Coverage
```powershell
# Run the complete test suite (250+ verified test cases across backend and testing):
.\backend\.venv\Scripts\python.exe -m pytest -v

# Run targeted test suites for Phase 1 - 4 architectural verifications:
.\backend\.venv\Scripts\python.exe -m pytest testing\test_phase1_phase2_architecture.py -v
.\backend\.venv\Scripts\python.exe -m pytest testing\test_phase2_dataset_lifecycle.py -v
.\backend\.venv\Scripts\python.exe -m pytest testing\test_deployment_gate.py -v
.\backend\.venv\Scripts\python.exe -m pytest testing\test_dl_model_registry.py -v
.\backend\.venv\Scripts\python.exe -m pytest testing\test_model_registry.py -v
.\backend\.venv\Scripts\python.exe -m pytest testing\test_multi_model_inference.py -v
.\backend\.venv\Scripts\python.exe -m pytest testing\test_live_federated_round.py -v
.\backend\.venv\Scripts\python.exe -m pytest backend\tests -v
```

### Frontend TypeScript Verification, Linting & Build
```powershell
cd frontend

# Verify TypeScript types without emitting code:
npx tsc --noEmit

# Run ESLint:
npm run lint

# Build production bundle:
npm run build
cd ..
```

### Python Linting with Ruff
```powershell
.\backend\.venv\Scripts\ruff.exe check backend/ ml/ dl/ testing/
```

---

## How to Retrain ML Models (Tabular Risk)

To re-run the nested cross-validation and 1-Standard-Error model selection pipeline:

```powershell
.\backend\.venv\Scripts\python.exe ml\training\train_risk_model.py
```

This pipeline executes:
1. Strict duplicate row filtering and class balance validation.
2. 50-fold nested repeated stratified cross-validation across Logistic Regression, Random Forest, and XGBoost.
3. 1,000-sample bootstrap confidence intervals for ROC-AUC, F1, MCC, Sensitivity, and Specificity.
4. Permutation significance testing (1,000 iterations).
5. 1-Standard-Error model selection, saving candidate artifacts to `ml/models/versions/` and updating `ml/models/registry_manifest.json`.

---

## How to Retrain DL Models (CT ResNet18)

### Running Leakage-Controlled Grouped Retraining
To train ResNet18 using perceptual cluster grouping ($pHash \le 2$) preventing near-duplicate slice leakage:

```powershell
.\backend\.venv\Scripts\python.exe dl\training\train_resnet18_audit.py
```
- Model checkpoints are written to `dl/models/ct/resnet18/best_checkpoint.pth`.
- Evaluation metrics and confusion matrices are written to `dl/models/ct/resnet18/metrics.json`.

---

## How to Run Federated Learning Simulations

To simulate decentralized multi-hospital training across 3 simulated hospitals (Apollo, Manipal, AIIMS):

```powershell
# Run 3-round local simulation with FedAvg aggregation:
.\backend\.venv\Scripts\python.exe dl\federated\simulate.py --rounds 3 --epochs 1 --mode iid
```
- Each client trains strictly on its isolated local partition (`dl/datasets/partitions/hospital_X`).
- Parameter updates are aggregated using FedAvg and saved as versioned checkpoints in `dl/models/federated/resnet18_fed_round_00X.pth`.
- Telemetry is persisted directly to `stonesense.db` (`federated_rounds` and `hospital_training_runs` tables).

---

## How to Add New Backend APIs

1. **Define Schema**: Create or extend a Pydantic model in `backend/app/schemas/` (e.g., `feature.py`).
2. **Create Router**: Implement your route handlers in `backend/app/api/v1/routes/` using FastAPI APIRouter:
   ```python
   from fastapi import APIRouter, Depends
   from app.core.auth import get_current_user
   from app.db.models import User

   router = APIRouter(prefix="/features", tags=["features"])

   @router.get("/")
   async def get_features(current_user: User = Depends(get_current_user)):
       return {"status": "ok"}
   ```
3. **Register Router in App Entrypoint**: Open [`backend/app/main.py`](backend/app/main.py), import your router, and register it inside `create_app()`:
   ```python
   from app.api.v1.routes.feature import router as feature_router
   # Inside create_app():
   application.include_router(feature_router, prefix="/api/v1")
   ```
   *(Note: Do NOT reference `backend/main.py`, which is an obsolete legacy file that has been removed.)*
4. **Write Tests**: Add corresponding test coverage in `testing/` or `backend/tests/`.

---

## How to Add New Frontend Pages & Routes

1. **Create Page Component**: Add a new React component in `frontend/src/pages/` (e.g., `NewAnalyticsPage.tsx`).
2. **Register Route in App Router**: Open [`frontend/src/App.tsx`](frontend/src/App.tsx) and declare the route:
   ```tsx
   import { NewAnalyticsPage } from './pages/NewAnalyticsPage';

   // Inside <Routes>:
   <Route
     path="/analytics"
     element={
       <ProtectedRoute allowedRoles={['hospital_user', 'developer', 'admin']}>
         <NewAnalyticsPage />
       </ProtectedRoute>
     }
   />
   ```
3. **Add Navigation Link**: If applicable, update the sidebar navigation in [`frontend/src/components/layout/Sidebar.tsx`](frontend/src/components/layout/Sidebar.tsx).
4. **Verify Type Correctness**: Run `npx tsc --noEmit` inside `frontend/` to ensure zero compilation errors.

---

## Troubleshooting & Common Issues

- **Port 8000 or 5173 Already in Use**:
  Run `stop_all.bat` or inspect port conflicts via PowerShell:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000, 5173 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  ```
- **"User account associated with this token was not found" (HTTP 401)**:
  Ensure `python -m backend.app.db.seed` has been executed to populate the `users` table in `stonesense.db`.
- **InconsistentVersionWarning on unpickling sklearn models**:
  This occurs when artifacts saved in scikit-learn 1.9.1 are loaded in 1.9.0. It is a non-blocking warning, but aligning package versions in the virtual environment resolves it.
