# StoneSense-AI System Architecture

**Document Version:** 2.0 (Verified Runtime Architecture)  
**Date:** October 2026  
**System Classification:** Explainable AI & Federated Decision-Support Research Platform  

---

## 1. Architectural Philosophy & Design Principles

StoneSense-AI is built on three foundational architectural principles:
1. **Decoupled Multi-Modal Evidence Streams**: Clinical urine biomarkers and axial CT scans are evaluated through completely separate, independent inference pipelines. There is **no synthetic early fusion** of modalities; clinicians evaluate both models' confidence and explainability independently.
2. **Zero Raw Patient Data Transfer (Federated Privacy)**: In the federated learning simulation, hospital nodes train exclusively on local partitions. Only model parameter matrices (NumPy weight tensors) are exchanged via Flower gRPC/WebSocket channels. Raw DICOM or JPEG image files never leave the simulated client boundary.
3. **Automated Governance & Deployment Gating**: No newly trained model checkpoint can be promoted to active clinician inference without passing a multi-metric automated deployment gate that verifies overall accuracy, recall thresholds on critical classes (`Stone`, `Tumor`), and regression tolerances against active models.

---

## 2. End-to-End System Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientTier["Client Layer (Vite + React 18 + TypeScript)"]
        HospUI["Hospital Console\n(/hospital-dashboard, /hospital/*)\nRole: hospital_user"]
        DevUI["Developer Console\n(/developer-dashboard)\nRole: developer, admin"]
        PubUI["Public Portal\n(/, /login, /privacy)"]
    end

    subgraph APITier["Application & Gateway Layer (FastAPI)"]
        CORS["CORS & Request Middleware"]
        AuthMid["JWT Auth & Scoping Dependency\n(decode_token, get_scoped_hospital_id)"]
        Router["APIRouter Gateway (52 Endpoints)"]
        WS["WebSocket Manager (ws_manager)\n/ws/federated"]
    end

    subgraph DataTier["Data & Persistence Layer"]
        DB[("Relational Store (stonesense.db)\nSQLite / PostgreSQL 16")]
        AuditStore[("Cryptographic Audit Table\n(SHA-256 Event Chain)")]
    end

    subgraph MLSubsystem["Tabular Risk Engine (ml/)"]
        MLReg["ModelRegistry\n(registry_manifest.json)"]
        ActiveML["Active Model:\nlogistic_regression_v001"]
        SHAP["Dynamic SHAP Explainer\n(LinearExplainer / TreeExplainer)"]
    end

    subgraph DLSubsystem["Axial CT Vision Engine (dl/)"]
        DLReg["DLModelRegistry"]
        ActiveDL["Active Model:\nResNet18 (Layer 4)"]
        GradCAM["Standard Grad-CAM Engine\n(Activations + Gradients)"]
        Wrappers["Experimental Wrappers:\nYOLO26, DINOv3, QKNN"]
    end

    subgraph FedSubsystem["Federated Learning & Gate Engine (dl/federated/ & backend/app/services/)"]
        Coord["FederatedCoordinator\n(Background Thread + Lock)"]
        Strategy["StoneSenseFedAvg\n(Custom Metric Aggregation)"]
        Gate["DeploymentGate Service\n(Threshold & Leakage Verification)"]
        Checkpoints["Checkpoint Store\n(dl/models/federated/*.pth)"]
    end

    %% Client to API
    PubUI -->|POST /auth/login| Router
    HospUI -->|REST /hospital/*, /predict/*| CORS
    DevUI -->|REST /developer/*| CORS
    HospUI <-->|WebSocket Real-Time Events| WS
    DevUI <-->|WebSocket Real-Time Events| WS

    %% Gateway to Auth & Router
    CORS --> AuthMid
    AuthMid --> Router
    Router --> DB
    Router --> AuditStore

    %% Router to Engines
    Router -->|POST /predict/risk| MLReg
    MLReg --> ActiveML
    ActiveML --> SHAP

    Router -->|POST /predict/image| DLReg
    DLReg --> ActiveDL
    ActiveDL --> GradCAM
    DLReg -.-> Wrappers

    %% Router to Federated & Governance
    Router -->|POST /developer/federated/rounds/start| Coord
    Coord --> Strategy
    Strategy --> Checkpoints
    Strategy --> DB
    Coord --> WS

    Router -->|POST /developer/model-versions/deploy| Gate
    Gate --> DB
    Gate --> AuditStore
```

---

## 3. Subsystem Breakdown

### 3.1 Presentation Tier (Frontend)
- **Framework**: React 18.3 with Vite 5.4 bundler and TypeScript 5.7.
- **Styling**: Tailwind CSS 3.4 with custom clinical palettes (`#F3F6F1` paper background, `#1F6F5C` clinical teal, and `#4F46E5` developer indigo).
- **Authentication Context**: [`AuthContext.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/context/AuthContext.tsx) manages JWT persistence, local storage tokens, role claims (`hospital_user`, `developer`, `admin`), and automated token refresh.
- **Hospital Scope Context**: [`HospitalContext.tsx`](file:///c:/Users/akhil/projects/StoneSense-AI/frontend/src/context/HospitalContext.tsx) synchronizes active hospital ID across the clinical console.

### 3.2 Gateway & API Tier (Backend)
- **Framework**: FastAPI 0.115+ running on Uvicorn ASGI server.
- **Entrypoint**: Factory function `create_app()` in [`backend/app/main.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/main.py).
- **Security & Authorization**:
  - `get_current_user` extracts Bearer JWT token, validates claims and expiration, and fetches user row from DB.
  - `get_scoped_hospital_id` ensures clinical users can only query or insert records belonging to their assigned hospital (`hospital_id`), preventing cross-tenant leakage.
  - Rate limiting enforced on login endpoints using in-memory sliding window limiter.

### 3.3 Relational Database Schema (`stonesense.db`)
Mapped using SQLAlchemy 2.0 declarative models in [`backend/app/db/models.py`](file:///c:/Users/akhil/projects/StoneSense-AI/backend/app/db/models.py):

| Table | Purpose | Key Attributes |
| :--- | :--- | :--- |
| `hospitals` | Tenant hospital registry | `id`, `hospital_code`, `name`, `current_model_version`, `dataset_size` |
| `users` | User credentials & roles | `id`, `email`, `password_hash`, `role`, `hospital_id`, `is_active` |
| `patients` | Patient clinical files | `id`, `hospital_id`, `name`, `age`, `urine_features`, `admitted_date` |
| `predictions` | Clinical predictions history | `id`, `hospital_id`, `patient_id`, `prediction_type`, `model_name`, `confidence` |
| `federated_rounds` | Global FL round telemetry | `round_number`, `global_accuracy`, `global_f1`, `global_loss`, `status` |
| `hospital_training_runs` | Per-hospital FL round metrics | `hospital_id`, `round_number`, `local_loss`, `local_accuracy`, `sample_count` |
| `model_versions` | Versioned models registry | `version_tag`, `model_family`, `accuracy`, `f1_score`, `is_deployed`, `gate_report` |
| `audit_logs` | Cryptographic event logs | `actor_user_id`, `action`, `resource_type`, `sha256_hash`, `timestamp` |
| `inference_logs` | Operational inference logs | `model_name`, `version_tag`, `confidence`, `latency_ms`, `timestamp` |
| `drift_records` | Data drift monitoring records | `metric_name`, `p_value`, `drift_detected`, `snapshot_date` |

### 3.4 Machine Learning Engine (Tabular Biomarkers)
- **Directory**: `ml/`
- **Data Source**: 79 cleaned observations with 6 urinary biomarkers (`gravity`, `ph`, `osmo`, `cond`, `urea`, `calc`).
- **Optimization**: 50-fold nested repeated stratified cross-validation (`StratifiedKFold(5)` $\times$ 10 repeats) executing hyperparameter search via `GridSearchCV`.
- **Selection**: 1-Standard-Error rule selects the simplest model within 1 SE of top performance. Logistic Regression v001 ($ROC\text{-}AUC = 0.8387$) is active.
- **Explainability**: [`ml/explainability/shap_explainer.py`](file:///c:/Users/akhil/projects/StoneSense-AI/ml/explainability/shap_explainer.py) dynamically inspects estimator attributes: uses `shap.LinearExplainer` for linear models and `shap.TreeExplainer` for tree ensembles.

### 3.5 Deep Learning Vision Engine (Axial CT Slices)
- **Directory**: `dl/`
- **Architecture**: ResNet18 convolutional neural network initialized with Torchvision ImageNet-1K pretrained weights. The final fully connected layer is adapted to 4 classes (`Cyst`, `Normal`, `Stone`, `Tumor`).
- **Split Protocol**: Perceptual cluster grouping ($pHash \le 2$ Hamming distance) on 12,446 slices prevents patient/slice leakage across train (8,708), validation (1,869), and test (1,869) partitions.
- **Explainability**: Standard Grad-CAM hooks capture forward activations and backward gradients from `model.layer4[-1]`, computing spatial weights via global average pooling and generating jet colormap overlays.

### 3.6 Federated Learning & Deployment Gate Engine
- **Decentralized Coordination**: Flower (`flwr`) framework using custom strategy [`StoneSenseFedAvg`](file:///c:/Users/akhil/projects/StoneSense-AI/dl/federated/strategy.py).
- **Weight Aggregation**: Parameters are extracted via `get_model_parameters` as NumPy float tensors, aggregated according to client sample count weights, and pushed back into the global model.
- **Deployment Gate Pipeline**:
  - Validates candidate metrics against production thresholds: Accuracy $\ge 95\%$, Macro F1 $\ge 94\%$, Stone Recall $\ge 96\%$, Tumor Recall $\ge 98\%$.
  - Enforces regression limit: Candidate cannot regress more than 2% relative to currently deployed active model.
  - Rejects unverified/leaky data: Checkpoints marked with `trained_on_leaky_partitions = True` are blocked from promotion.
