# StoneSense-AI Developer Guide

Welcome to the **StoneSense-AI Developer Guide**. This document provides detailed, technical guidance for setup, daily workflows, automation scripts, model retraining, API extension, and troubleshooting.

---

## Table of Contents
1. [Complete Setup Guide](#complete-setup-guide)
2. [Batch File Usage](#batch-file-usage)
3. [PowerShell Launcher Usage](#powershell-launcher-usage)
4. [Project Workflow](#project-workflow)
5. [How to Retrain Models](#how-to-retrain-models)
6. [How to Rerun SHAP](#how-to-rerun-shap)
7. [How to Rerun Grad-CAM](#how-to-rerun-grad-cam)
8. [How to Add New APIs](#how-to-add-new-apis)
9. [How to Add New Frontend Pages](#how-to-add-new-frontend-pages)
10. [Git Workflow & Release Checklist](#git-workflow--release-checklist)
11. [Common Debugging & Terminal Commands](#common-debugging--terminal-commands)
12. [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

---

## Complete Setup Guide

### System Requirements
- **OS**: Windows 10/11
- **Python**: 3.10+ (Added to system PATH)
- **Node.js**: 18+ & npm 9+
- **Git**: 2.30+

### Automated Setup
To configure all virtual environments (`backend\.venv`, `ml\.venv`, `dl\.venv`) and frontend Node modules in a single step:

```cmd
install_all.bat
```

---

## Batch File Usage

The project includes pre-built batch scripts in the repository root:

- `developer_launcher.bat`: Launch the interactive console menu.
- `run_all.bat`: Concurrently launch Backend and Frontend in separate windows and open Web UI + Swagger docs.
- `run_backend.bat`: Start FastAPI server (`http://127.0.0.1:8000`).
- `run_frontend.bat`: Start Vite React dev server (`http://localhost:5173`).
- `run_ml.bat`: Launch ML interactive menu.
- `run_dl.bat`: Launch DL interactive menu.
- `stop_all.bat`: Safely terminate running processes (`node.exe`, `python.exe`, `uvicorn`).

---

## PowerShell Launcher Usage

For PowerShell users, `developer_launcher.ps1` provides additional automation features including colored logging, automated log rotation in `logs/`, and port collision handling.

To run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process
.\developer_launcher.ps1
```

---

## Project Workflow

### Recommended Daily Workflow
1. Pull latest changes: `git pull origin main`
2. Launch full application: `run_all.bat`
3. Develop features in `backend/` or `frontend/`
4. Validate changes:
   - Backend: `pytest` in `backend/`
   - Frontend: `npx tsc --noEmit` in `frontend/`
5. Commit & push: `git push origin main`

---

## How to Retrain Models

### Retraining ML XGBoost Model
1. Open `run_ml.bat` or run:
   ```cmd
   call ml\.venv\Scripts\activate
   python ml\training\train_risk_model.py
   ```
2. Saved artifacts are written to `ml/models/` and `ml/outputs/`.

### Retraining DL ResNet18 Model
1. Open `run_dl.bat` or run:
   ```cmd
   call dl\.venv\Scripts\activate
   python dl\training\train_resnet18.py
   ```
2. Model checkpoints are saved to `dl/models/best_resnet18.pth`.

---

## How to Rerun SHAP

To recompute SHAP feature attributions on tabular dataset test splits:
```cmd
call ml\.venv\Scripts\activate
python ml\explainability\shap_explainer.py
```
Output charts will be written to `ml/outputs/shap_summary.png` and `ml/outputs/shap_waterfall.png`.

---

## How to Rerun Grad-CAM

To generate visual saliency heatmaps for CT scan image slices:
```cmd
call dl\.venv\Scripts\activate
python dl\explainability\gradcam.py
```
Output heatmaps will be generated in `dl/outputs/gradcam_overlays/`.

---

## How to Add New APIs

1. Create a new Pydantic schema in `backend/app/schemas/`.
2. Create a router file in `backend/app/api/v1/routes/` (e.g., `new_feature_router.py`).
3. Define route handlers using standard FastAPI decorators (`@router.post()`, `@router.get()`).
4. Register the router in `backend/main.py`:
   ```python
   app.include_router(new_feature_router.router, prefix="/api/v1")
   ```
5. Run backend tests to verify contract stability.

---

## How to Add New Frontend Pages

1. Create a view component in `frontend/src/pages/` (e.g., `AnalyticsPage.tsx`).
2. Update navigation routes in `frontend/src/App.tsx`:
   ```tsx
   <Route path="/analytics" element={<AnalyticsPage />} />
   ```
3. Add a navigation item in `frontend/src/components/Navbar.tsx`.
4. Run `npx tsc --noEmit` inside `frontend/` to confirm zero TypeScript errors.

---

## Git Workflow & Release Checklist

### Git Branching
- `main`: Production-ready releases.
- `feature/<name>`: New feature implementations.
- `fix/<name>`: Bug fixes and repairs.

### Release Checklist
- [ ] `npx tsc --noEmit` passes without errors in `frontend/`.
- [ ] `npm run build` generates `dist/` successfully.
- [ ] `pytest` passes cleanly in `backend/`.
- [ ] Model artifacts exist in `ml/models/` and `dl/models/`.
- [ ] Documentation updated in `README.md`.

---

## Common Debugging & Terminal Commands

### Frontend Type Check & Build
```cmd
cd frontend
npx tsc --noEmit
npm run build
```

### Backend Test Execution
```cmd
cd backend
..\backend\.venv\Scripts\pytest
```

### Port Conflict Check (Windows)
```cmd
netstat -ano | findstr :8000
netstat -ano | findstr :5173
```

---

## Frequently Asked Questions (FAQ)

**Q: Why does the browser show a blank page on dev start?**  
A: Ensure TypeScript errors are resolved (`npx tsc --noEmit`) and Vite plugin imports match Vite 5 syntax in `vite.config.ts`.

**Q: How do I change the backend server port?**  
A: Update the `--port` flag in `run_backend.bat` and update `VITE_API_BASE_URL` in `frontend/src/services/api.ts`.

**Q: Where are training reports saved?**  
A: Machine Learning reports are saved in `ml/outputs/` and Deep Learning reports in `dl/outputs/`.
