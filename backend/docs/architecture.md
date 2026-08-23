# Trustworthy Backend Architecture & Pipelines

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
              │    Prediction Routes      │
              └──────┬─────────────┬──────┘
                      │             │
         ┌──────────▼──┐       ┌──▼──────────┐
         │  Prediction │       │ModelLoader  │
         │   Service   │       │(Singleton)  │
         └─────────────┘       └─────────────┘
```

### 1. Singleton Model Loading (`core/startup.py`)

Both the ML (XGBoost) and DL (ResNet18) models, along with the pre-fitted preprocessing pipeline and mappings, are loaded once into RAM during FastAPI startup. This prevents reloading overheads per HTTP request.

### 2. Independent Explainability (`services/explainability_service.py`)

SHAP and Grad-CAM explain the respective standalone prediction outputs. The clinical and imaging workflows remain separate because their datasets are not patient-level paired.
