# Standalone Inference Flow

This document details the standalone clinical-risk and CT-image inference flows.

---

## Inference Diagram

```
[FastAPI Request Client]
  │
  ├──► 1. POST /api/v1/predict/risk or /api/v1/predict/image
  │      ├─ JSON: PatientInformation for clinical risk
  │      └─ File: image for CT classification
  │
  ├──► 2. Preprocessing Utilities
  │      ├─ Convert raw bytes to standard torch.Tensor
  │      └─ Map clinical dict keys to tabular formats
  │
  ├──► 3. ResNet18 & Grad-CAM (DL, image flow)
  │      ├─ Run forward pass on tensor to classify CT condition
  │      └─ Calculate final layer activation map and write overlay image
  │
  ├──► 4. XGBoost & SHAP (ML, clinical flow)
  │      ├─ Run tabular pipeline to estimate stone risk probability
  │      └─ Run TreeExplainer to calculate local feature contributions
  │
  └───◄ 5. Return the selected standalone model output and explanation
```
