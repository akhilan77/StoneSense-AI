# API Endpoints Reference Documentation (v1)

FastAPI endpoints exposed under root path prefix `/api/v1/` for StoneSense-AI.

---

## 1. GET /api/v1/health

Returns server status indicators, version values, and pre-loaded model checks.

### Response Example

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "resnet18_ct_classification": true,
    "xgboost_risk_prediction": true,
    "preprocessing_pipeline": true
  },
  "hardware": {
    "cuda_available": false,
    "active_device": "cpu"
  }
}
```

---

## 2. POST /api/v1/predict/risk

Calculates patient-level risk metrics from clinical parameters.

### Request Body (`PatientInformation`)

```json
{
  "age": 45,
  "gender": "male",
  "bmi": 24.5,
  "blood_pressure": 120.0,
  "diabetes": false,
  "family_history": true,
  "water_intake": 2.5,
  "urine_ph": 6.0,
  "urine_specific_gravity": 1.015,
  "calcium": 4.5,
  "uric_acid": 3.0,
  "creatinine": 1.0
}
```

### Response Example

```json
{
  "probability": 0.2524,
  "risk_level": "Low",
  "confidence": 0.95,
  "inference_time_sec": 0.003
}
```

---

## 3. POST /api/v1/predict/image

Performs classification on uploaded CT scan slice.

### Request

- **Header:** `Content-Type: multipart/form-data`
- **Body:** `image` (binary file)

### Response Example

```json
{
  "class_name": "Cyst",
  "confidence": 0.4717,
  "inference_time_sec": 0.035
}
```

---
