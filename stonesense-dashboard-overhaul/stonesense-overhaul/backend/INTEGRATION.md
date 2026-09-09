# Backend integration — Developer + Hospital dashboards

## 1. Copy files into your repo
```
backend/app/db/models.py           (new)
backend/app/db/database.py         (new)
backend/app/db/seed.py             (new)
backend/app/schemas/dashboard.py   (new)
backend/app/api/v1/routes/hospital.py    (new)
backend/app/api/v1/routes/developer.py   (new)
```

## 2. Add dependency
```
pip install sqlalchemy
```
Add `sqlalchemy` to `backend/requirements.txt`.

## 3. Wire into `main.py`
```python
from app.db.database import init_db
from app.api.v1.routes import hospital, developer

app.include_router(hospital.router, prefix="/api/v1/hospital", tags=["hospital"])
app.include_router(developer.router, prefix="/api/v1/developer", tags=["developer"])

@app.on_event("startup")
def on_startup():
    init_db()
```

## 4. Seed demo data (once)
```
cd backend
python -m app.db.seed
```
Creates 3 demo hospitals, model version history, sample logs and drift points —
enough for both dashboards to render real data immediately.

## 5. Tag your existing predict endpoints with hospital_id
Your current `POST /api/v1/predict/risk` and `POST /api/v1/predict/image`
(in `routes/predict.py`) don't change their ML/DL logic at all. Add:

1. `hospital_id: int` to the request (form field for image, body field for risk).
2. After computing the result, persist one row:

```python
from app.db.database import get_db
from app.db.models import Prediction
import time

# inside the existing handler, after you have `result` and `confidence`:
start = ...  # you likely already time this for your latency benchmark
prediction = Prediction(
    hospital_id=hospital_id,
    prediction_type="risk",              # or "image"
    model_name="xgboost_risk_v1_1",       # or your resnet tag
    result_label=str(result.risk_level),  # or class_name
    confidence=confidence,
    latency_ms=(time.perf_counter() - start) * 1000,
)
db.add(prediction)
db.commit()
```

That's the entire integration — the developer dashboard's aggregates
(`/api/v1/developer/system-monitoring`, etc.) read from this same
`predictions` table, so once hospital requests start writing to it,
the developer console updates automatically.

## 6. Data-isolation guarantee
- Every hospital-facing query filters `WHERE hospital_id = :id`.
- Every developer-facing query reads `model_versions`, `hospital_update_logs`,
  `system_logs`, `drift_records`, or does `COUNT`/`AVG` over `predictions` —
  it never selects a `Patient` row or `urine_features`. This matches the
  "independently sourced, not patient-paired" limitation already documented
  in your README: the developer console reasons about model/system health,
  never about an individual patient or hospital's raw clinical data.
- When you add real auth later, replace the `hospital_id` request param with
  a value read out of the JWT — no query logic changes.
