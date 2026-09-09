# StoneSense-AI — Developer & Hospital Dashboard Overhaul

This bundle adds the two dashboards from your wireframes on top of the
existing FastAPI + React StoneSense-AI stack, with real multi-hospital
data isolation (no auth yet — hospital selected via dropdown, as agreed).

## Quick start
1. Open **`preview/dashboards_preview.html`** directly in a browser first —
   no build step. It's a static visual reference for both dashboards with
   the role-switch toggle, so you can sanity-check the look before wiring
   real data.
2. Follow **`backend/INTEGRATION.md`** to drop the new DB models/routes into
   your FastAPI backend and seed demo data.
3. Follow **`frontend/INTEGRATION.md`** to drop the new pages/context/theme
   into your Vite/React frontend.

## What's new

**Backend**
- SQLite persistence: `hospitals`, `patients`, `predictions`, `model_versions`,
  `hospital_update_logs`, `system_logs`, `drift_records`
- `/api/v1/hospital/*` — hospital-scoped, filtered by `hospital_id` on every query
- `/api/v1/developer/*` — cross-hospital aggregates only (model performance,
  deployment control, federated update history, system monitoring, drift) —
  never reads raw patient rows, matching your existing dataset-limitation note

**Frontend**
- `HospitalContext` — app-wide hospital selector (dropdown, no auth)
- `HospitalDashboard` — stepped clinical workflow: input → ML/DL results →
  explainability (SHAP/Grad-CAM) → trustworthy assessment, plus a recent-cases
  panel, all scoped to the selected hospital
- `DeveloperDashboard` — model performance/deploy table, federated update
  history, live system-monitoring stats, drift snapshots
- New sidebar-based `AppLayout`, replacing the old top-nav — existing
  `/`, `/risk-prediction`, `/stone-detection` pages stay intact and are
  reachable from the Hospital console's sidebar
- New palette/type system (see `frontend/INTEGRATION.md` §2) shared by
  both dashboards, with a distinct indigo accent on the developer side to
  visually separate the two roles

## Data-isolation model (your "needs real separation logic" requirement)
Every hospital-facing endpoint takes `hospital_id` and filters on it in SQL.
Every developer-facing endpoint only ever touches `model_versions`, log/drift
tables, or `COUNT`/`AVG` aggregates over `predictions` — it has no code path
that can select a `Patient` row. When you add real auth later, `hospital_id`
moves from an explicit request param to a JWT claim; no query changes.
