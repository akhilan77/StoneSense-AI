# StoneSense AI

StoneSense AI is an explainable AI-based kidney stone detection and risk prediction system designed for modular, production-oriented development. This release focuses on clean architecture, API contract design, and mock response scaffolding so that Day 2 development can proceed with a stable interface boundary.

## Folder structure

```text
backend/
  app/
    api/
      v1/
        routes/
    core/
    models/
    schemas/
    services/
    utils/
    config/
  tests/
  main.py
  requirements.txt

frontend/
  src/
    components/
    pages/
    layouts/
    services/
    hooks/
    assets/
    types/
    styles/

ml/
  datasets/
  notebooks/
  preprocessing/
  training/
  inference/
  models/
  utils/

dl/
  datasets/
    train/
    val/
    test/
  preprocessing/
  training/
  inference/
  models/
  utils/

infra/
  docker/
  nginx/
  scripts/
```

## Project architecture

The repository follows a layered, clean-architecture pattern:

- Presentation layer: React + Vite frontend and FastAPI route handlers.
- Application layer: service objects that orchestrate contract-level logic.
- Domain layer: Pydantic schemas and typed request/response entities.
- Infrastructure layer: configuration, environment setup, and future ML/DL runtime integrations.

The design aligns with SOLID principles by keeping routes thin, services focused, and data contracts explicit.

## API endpoints

### Core system endpoints

- `GET /`
  - Returns a service identity message.
- `GET /health`
  - Returns a readiness payload.

### Risk prediction

- `POST /predict/risk`
  - Accepts `PatientInformation` and returns a mock `RiskPredictionResponse`.

### Image detection

- `POST /predict/image`
  - Accepts an image upload and returns a mock `StoneDetectionResponse`.

### Final assessment

- `POST /assess`
  - Returns a single composite assessment object containing:
    - `patient`
    - `risk_prediction`
    - `stone_detection`
    - `explainability`
    - `recommendation`

## Development workflow

1. Install backend dependencies from `backend/requirements.txt`.
2. Start the FastAPI server with Uvicorn from the backend package root.
3. Install frontend dependencies with npm from `frontend/`.
4. Keep model schemas, TypeScript types, and service contracts synchronized.
5. Introduce real ML/DL implementations only after the public contract is stable.

## Notes

- No ML or DL inference is implemented in this release.
- All endpoints currently return deterministic mock JSON to support early integration and validation.
- The architecture is intentionally prepared for future SHAP, LIME, Grad-CAM++, and GenAI integrations.
