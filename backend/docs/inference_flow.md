# Assessment Inference Flow

This document details the multi-modal endpoint lifecycle flow when `/api/v1/assessment` receives image and patient metadata inputs.

---

## Inference Diagram

```
[FastAPI Request Client]
  │
  ├──► 1. POST /api/v1/assessment
  │      ├─ Form-data: patient_data (JSON)
  │      └─ File: image (uploaded bytes)
  │
  ├──► 2. Preprocessing Utilities
  │      ├─ Convert raw bytes to standard torch.Tensor
  │      └─ Map clinical dict keys to tabular formats
  │
  ├──► 3. ResNet18 & Grad-CAM (DL)
  │      ├─ Run forward pass on tensor to classify CT condition
  │      └─ Calculate final layer activation map and write overlay image
  │
  ├──► 4. XGBoost & SHAP (ML)
  │      ├─ Run tabular pipeline to estimate stone risk probability
  │      └─ Run TreeExplainer to calculate local feature contributions
  │
  └───◄ 5. Return Unified JSON Response
```
