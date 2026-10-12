# StoneSense-AI Backend Subsystem

The `backend/` directory contains the FastAPI REST & WebSocket application, relational database persistence layer, security middleware, and model inference services.

---

## Architecture & Directory Layout

```text
backend/
├── alembic/                  # Database migration scripts and schema versions
│   └── versions/             # Reversible migration revisions
├── app/
│   ├── api/v1/routes/        # Modular API routes
│   │   ├── admin.py          # Audit log retrieval and administrator endpoints
│   │   ├── auth.py           # JWT authentication, login rate limiting, token refresh
│   │   ├── developer.py      # Cross-hospital developer console endpoints (stats, gate, deploy, rollback)
│   │   ├── health_router.py  # Health probe endpoint (/api/v1/health)
│   │   ├── hospital.py       # Hospital-scoped clinical workflows, history, datasets
│   │   ├── model_router.py   # Model metadata and status endpoints (/api/v1/models)
│   │   ├── patients.py       # Patient record management (scoped by hospital_id)
│   │   ├── predict_router.py # Inference & concordance routes (/predict/risk, /predict/image, /predict/concordance)
│   │   └── root_router.py    # Root greeting endpoint (/)
│   ├── config/               # Settings configuration (settings.py)
│   ├── core/                 # Auth dependencies, security utilities, startup routines
│   ├── db/                   # SQLAlchemy database engine, session factory, models.py, seed.py
│   ├── schemas/              # Typed Pydantic request and response schemas (concordance.py, responses.py)
│   ├── services/             # Core business logic
│   │   ├── concordance_service.py # Pure, deterministic evidence concordance engine
│   │   ├── deployment_gate.py # Multi-metric validation and rollback engine
│   │   ├── explainability_service.py # Grad-CAM and SHAP generation
│   │   ├── federated_coordinator.py  # Multi-hospital FL coordinator
│   │   ├── model_loader.py   # Runtime model artifact loader
│   │   ├── prediction_service.py # Unified inference gateway
│   │   └── ws_manager.py     # Real-time WebSocket connection manager
│   ├── utils/                # Preprocessing, bounded upload read, and content validation helpers
│   └── main.py               # Main application factory: create_app()
└── tests/                    # Unit & API tests for endpoints, auth, concordance, and image upload hardening
```

---

## Configuration & Environment Variables

All safety thresholds are consolidated in [`app/config/settings.py`](app/config/settings.py):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CONCORDANCE_RISK_THRESHOLD` | `0.50` | Probability cutoff above which urine risk is elevated |
| `CONCORDANCE_TABULAR_MIN_CONFIDENCE` | `0.60` | Minimum confidence for tabular risk model |
| `CONCORDANCE_IMAGING_MIN_CONFIDENCE` | `0.60` | Minimum confidence for CT imaging model |
| `CONCORDANCE_DISCLAIMER` | *(Standard text)* | Regulatory disclaimer on unpaired training datasets |
| `CT_UPLOAD_MAX_BYTES` | `10485760` (10 MB) | Bounded upload size limit (HTTP 413) |
| `CT_UPLOAD_MAX_PIXELS` | `25000000` (25 MP) | Decompression bomb protection limit (HTTP 413) |
| `CT_UPLOAD_MIN_DIMENSION` | `32` | Minimum image dimension in pixels (HTTP 422) |
| `CT_OOD_MIN_CONFIDENCE` | `0.50` | Heuristic gate minimum confidence (status='needs_review') |
| `CT_OOD_MAX_ENTROPY` | `1.30` | Heuristic gate maximum predictive entropy in nats |
| `CT_OOD_MIN_STD` | `8.0` | Minimum pixel standard deviation for uniform/blank image warnings |
| `CT_OOD_MAX_ASPECT_RATIO` | `3.5` | Maximum aspect ratio before non-axial rejection (HTTP 422) |

---

## Application Entrypoint & Startup

The backend is initialized via the factory `create_app()` in [`app/main.py`](app/main.py).  
*(Note: A legacy root-level `backend/main.py` stub was deleted in commit `5a64a58`.)*

### Starting Locally (Development)
```bash
# From within the backend/ directory:
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload

# Or from repository root:
.\run_backend.bat
```

- **Base URL:** `http://127.0.0.1:8000`
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **Health Check:** `GET /api/v1/health`

---

## Database & Migrations

The backend defaults to SQLite (`stonesense.db` at repo root) or connects to PostgreSQL via `STONESENSE_DATABASE_URL`:

```bash
# Run migrations:
alembic upgrade head

# Seed initial demonstration accounts:
python -m backend.app.db.seed
```

Demo Accounts:
- **Hospital User:** `dr.aisha@apex-health.org` / `HospitalPass123!`
- **Developer:** `dev@stonesense.ai` / `DevPass123!`
- **Admin:** `admin@stonesense.ai` / `AdminPass123!`

---

## Running Tests
```bash
pytest backend/tests -v
```
