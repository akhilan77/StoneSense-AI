# End-to-End System Validation Report

- **Date:** 2026-08-05
- **Assessor:** Senior QA Automation Engineer
- **Target Application:** StoneSense-AI (React + FastAPI + PyTorch/XGBoost)

---

## 1. End-to-End Workflow Status

| Test ID | Workflow                | Input Scenario                               | Expected Output                                              | Status  |
| ------- | ----------------------- | -------------------------------------------- | ------------------------------------------------------------ | ------- |
| VAL-01  | Server Health check     | `GET /api/v1/health`                         | HTTP 200, models loaded is `true`                            | ✅ PASS |
| VAL-02  | Model Info query        | `GET /api/v1/models`                         | HTTP 200, specifications match                               | ✅ PASS |
| VAL-03  | Tabular Risk prediction | Balanced Patient Profile                     | Low/High risk prediction probability                         | ✅ PASS |
| VAL-04  | CT Image classification | Valid grayscale CT slice                     | Target labels (Normal/Cyst/Stone/Tumor)                      | ✅ PASS |
| VAL-05  | Trustworthy Assessment  | Optional clinical profile + optional CT file | Separate evidence, SHAP, Grad-CAM URL, and rule-based status | ✅ PASS |
| VAL-06  | Client compile check    | `npm run build`                              | Succeeded with 0 errors                                      | ✅ PASS |

---

## 2. Invalid Input Boundary Evaluations

- **Empty Patient Biomarkers:** Handled gracefully via FastAPI Pydantic validator checks, raising `422 Unprocessable Entity` when required keys are missing.
- **Unsupported Image Formats:** Routes reject non-supported files (e.g., text, PDF) with standard validation messages.
- **Offline Backend recovery:** Frontend state sets descriptive error messages in red cards if backend endpoints throw connection refuse errors, preventing UI crashes.
