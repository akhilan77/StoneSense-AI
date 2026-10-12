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
│   │   ├── predict_router.py # Real inference routes (/predict/risk, /predict/image)
│   │   └── root_router.py    # Root greeting endpoint (/)
│   ├── config/               # Settings configuration (settings.py)
│   ├── core/                 # Auth dependencies, security utilities, startup routines
│   ├── db/                   # SQLAlchemy database engine, session factory, models.py, seed.py
│   ├── schemas/              # Typed Pydantic request and response schemas
│   ├── services/             # Core business logic
│   │   ├── deployment_gate.py # Multi-metric validation and rollback engine
│   │   ├── explainability_service.py # Grad-CAM and SHAP generation
│   │   ├── federated_coordinator.py  # Multi-hospital FL coordinator
│   │   ├── model_loader.py   # Runtime model artifact loader
│   │   ├── prediction_service.py # Unified inference gateway
│   │   └── ws_manager.py     # Real-time WebSocket connection manager
│   ├── utils/                # Preprocessing and image conversion helpers
│   └── main.py               # Main application factory: create_app()
└── tests/                    # Unit tests for API endpoints and auth dependencies
```

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
