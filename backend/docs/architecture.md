# Unified Backend Architecture & Pipelines

This backend utilizes a clean architecture separating HTTP routing boundaries, singleton resource loading, and preprocessing routines from core AI models.

---

## Architecture Grid

```
             FastAPI Client HTTP Requests
                          │
            ┌─────────────▼─────────────┐
            │       API v1 Routes       │
            └─────────────┬─────────────┘
                          │ (Orchestrates Services)
            ┌─────────────▼─────────────┐
            │    AssessmentService      │
            └──────┬─────────────┬──────┘
                   │             │
        ┌──────────▼──┐       ┌──▼──────────┐
        │  Prediction │       │ModelLoader  │
        │   Service   │       │(Singleton)  │
        └─────────────┘       └─────────────┘
```

### 1. Singleton Model Loading (`core/startup.py`)
Both the ML (XGBoost) and DL (ResNet18) models, along with the pre-fitted preprocessing pipeline and mappings, are loaded once into RAM during FastAPI startup. This prevents reloading overheads per HTTP request.

### 2. Multi-Modal Assessment Orchestration (`services/assessment_service.py`)
Aggregates predictions from both pipelines, overlays Grad-CAM visual heatmaps, calculates local SHAP feature contributions, and formats clinical recommendations into one unified JSON payload.
