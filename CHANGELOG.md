# Changelog

All notable changes to the StoneSense-AI project are documented in this file based on repository history.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### NHANES v003b: LightGBM early-stop fix, class-imbalance comparison, accuracy-focused operating point (pending review, not deployed)
- **LightGBM:** `best_iteration_` 3 -> 190. Cause (computed in notebook 04): the objective's default `binary_logloss` was evaluated next to AUC, early stopping stopped on the first metric to stall, and `scale_pos_weight` made unweighted validation logloss rise immediately. Fix: stop on AUC only (`first_metric_only=True`, patience 200), conservative parameters, one imbalance strategy, early-stopping half separated from the calibration half.
- **Imbalance:** every model notebook compares none / class weights / random oversampling / random undersampling / SMOTENC in an `imblearn` Pipeline (training rows only; SMOTENC with declared categorical columns), calibrates each on the real-prevalence validation half and selects on validation only. `imbalanced-learn` added to `requirements-ml.txt`.
- **Operating points:** `accuracy` (max sensitivity s.t. validation accuracy >= 72%) and `balanced` (max balanced accuracy), both stored in `registry_manifest.json` with validation and test metrics. Test accuracy 71.6% to 72.7% at the accuracy point, at sensitivity 48.2% to 50.4%.
- **Notebooks:** one `show_metrics_table` helper prints the full metric set (accuracy included) at every stage; per-fold LOCO accuracy / sensitivity; provenance (notebook, timestamp, device, training time, library versions, split hash) in every metrics JSON; notebook 05 fails loudly on missing / mismatched artifacts.
- **Backend / frontend:** compare endpoint returns both operating points per model and accepts `operating_point`; the four-model table has an operating-point toggle (accuracy-focused default) with sensitivity next to accuracy and a caution line. Models registered as `*_nhanes_v003b`.
- **Tests:** `testing/test_nhanes_v003b.py`, `testing/test_nhanes_notebooks.py`; `backend/tests/test_nhanes_compare.py` updated.

### NHANES v003: four-model clinical-profile comparison (pending review, not deployed)
- **Cohort v2:** `ml/preprocessing/add_history_features.py` adds self-reported history features (`BPQ020`, `BPQ080`, `DIQ010`, `MCQ080` usable; `MCQ520`/`MCQ550` excluded because they exist only in 2017-2018) by a one-to-one merge, drops `KIQ029`, and writes `history_feature_availability.csv`. `ml/preprocessing/extract_nhanes_zip.py` extracts the raw zip safely.
- **Notebooks:** `ml/notebooks/nhanes_v003/00-05` (+ `04b` logistic-regression reference) train EBM, CatBoost, XGBoost and LightGBM on one frozen, hash-checked cycle-based split; shared code in `ml/training/nhanes_common.py`; run with `python ml/training/run_nhanes_notebooks.py`. Each model's test cycle is scored once (ledger-enforced).
- **Registry:** `CatBoostRiskModel`, `LightGBMRiskModel`, `EBMRiskModel`, `XGBoostNhanesRiskModel` wrappers around a self-contained `NhanesBundle`; `*_nhanes_v003` entries added as `pending_review`. `get_model_by_family` keeps preferring legacy models.
- **Backend:** `POST /api/v1/predict/risk/compare` and `GET /api/v1/predict/risk/compare/spec` (inputs validated against `feature_spec.json`, not persisted, no migration).
- **Frontend:** "Clinical profile (NHANES)" mode on the risk page and in the dashboard workflow, built from `feature_spec.json`, with each model's held-out test metrics and the majority-class baseline.
- **Dependencies:** `catboost`, `lightgbm`, `interpret` pinned; notebook tooling in `requirements-ml.txt`.

## [1.2.0] - 2026-10-12

### End-to-End Hardening & Security Release
- **Concordance Safety Fixes (Phase 1)**:
  - Renamed `CONCORDANT_LOW_CONCERN` to `CONCORDANT_NO_STONE_EVIDENCE` ("No stone evidence in either stream"). Explanation explicitly clarifies this is NOT a clinical clearance, citing audited CT slice stone sensitivity (65.70%, scan 80.95%) and small tabular sample size ($N=79$).
  - Deprecated and disabled client-supplied raw probability payloads by default (`allow_direct_concordance_payload=False`); endpoint resolves server-stored prediction IDs (`risk_prediction_id`, `image_prediction_id`, `patient_id`) with strict hospital tenant and patient scoping (403 on mismatch).
  - Maintained safe backward-compatible deserialization for legacy stored assessment records.
  - Documented audit trail accurately as standard relational database rows without cryptographic claims, ensuring zero PII logging.
- **Upload Hardening Follow-ups (Phase 2)**:
  - Scoped Pillow decompression bomb checks to fail at `MAX_IMAGE_PIXELS` (promoting `DecompressionBombWarning` to an error) and pre-checking raw dimensions, restoring global settings in `finally`. Verified rejection of images between 1x and 2x pixel limits.
  - Configured Nginx ingress proxy `client_max_body_size 10M;` consistent with backend limits.
  - Built empirical OOD evaluation script (`testing/eval_ood_gate.py`), reporting metrics across 100 in-distribution CT slices (0.0% rejected) and 125 synthetic OOD patterns (14.4% rejected; Gaussian noise fooled ResNet18 with 99.98% confidence). Documented that gate is an empirical heuristic.
- **Test Side-Effects & Artifact Isolation (Phase 3)**:
  - Eliminated test side-effects on tracked model checkpoints (`latest.pth`) by isolating federated training checkpoints to temporary process directories via `STONESENSE_MODELS_DIR`.
  - Rebound test databases to isolated temporary SQLite instances (`stonesense_test_<pid>.db`), guaranteeing that `stonesense.db` and git-tracked files remain completely untouched during test runs.
- **Frontend Lint & Architecture Decoupling (Phase 4)**:
  - Resolved all Fast Refresh (`react-refresh/only-export-components`) and `react-hooks/exhaustive-deps` warnings.
  - Decoupled `AuthContext` and `HospitalContext` into dedicated context definitions (`authContextDef.ts`, `hospitalContextDef.ts`) and hooks (`useAuth.ts`, `useHospital.ts`). ESLint now passes cleanly with 0 errors and 0 warnings under `--max-warnings 0`.
- **Patient PII Authenticated Encryption at Rest (Phase 5)**:
  - Implemented column-level authenticated encryption (AES-256-GCM with fresh 96-bit nonces) via SQLAlchemy `EncryptedString` TypeDecorator for `patients.name`, `patients.phone`, and `patients.email`.
  - Added keyed HMAC-SHA256 blind indexing (`name_bidx`, `phone_bidx`, `email_bidx`) enabling deterministic exact-match querying without exposing plaintext.
  - Configured fail-closed key management requiring `STONESENSE_PII_KEYS` in non-development environments, with support for zero-downtime key rotation.
  - Authored Alembic migration (`0002_patient_pii_encryption_and_blind_indexes`) and idempotent backfill script (`backfill_encrypt_patients.py`) with automatic timestamped database backup.
- **Label Normalization & Route Sanitization (Phase 6)**:
  - Normalized all user-facing, component, and doc mentions of "Grad-CAM++" to "Grad-CAM".
  - Moved LIME mentions to "Not implemented / future work" section.
  - Fixed broken `/home` dashboard cards by linking to valid existing routes (`/hospital/federated-learning`, `/hospital/private-dataset`) and adding `/hospital-dashboard` fallback redirects for `/risk-prediction` and `/stone-detection`.
  - Archived legacy notes into `docs/archive/`.

---

## [1.1.0] - 2026-10-12

### Clinical-Safety & Endpoint Hardening Fixes
- **Evidence Concordance Engine**:
  - Implemented pure, deterministic, rule-based evidence concordance service (`evaluate_concordance`) in `backend/app/services/concordance_service.py`.
  - Defined 6 closed outcome codes: `CONCORDANT_STONE_EVIDENCE`, `CONCORDANT_LOW_CONCERN`, `DISCORDANT`, `NON_STONE_FINDING`, `LOW_CONFIDENCE`, and `INSUFFICIENT_DATA` with strict precedence hierarchy (`INSUFFICIENT_DATA` > `NON_STONE_FINDING` > `LOW_CONFIDENCE` > `CONCORDANT_STONE_EVIDENCE` > `CONCORDANT_LOW_CONCERN` > `DISCORDANT`).
  - Enforced mandatory clinician review escalation for any CT non-stone incidental finding (Cyst or Tumor) regardless of tabular risk.
  - Exposed authenticated endpoint `POST /api/v1/predict/concordance` returning full input audit, applied thresholds, triggered rules, and regulatory decision-support disclaimers.
  - Replaced hardcoded "Trustworthy assessment" numbers ("71%", "94.2%") in `HospitalDashboard.tsx` with live dynamic evaluation component (`ConcordanceCard.tsx`) providing color+icon indicators and empty-state guidance.
- **CT Upload Endpoint Hardening & Out-of-Distribution (OOD) Gate**:
  - Enforced bounded request stream reading (64 KB chunks up to configurable 10 MB limit) with HTTP 413 responses for oversized payloads.
  - Added strict magic-bytes validation restricting uploads exclusively to PNG and JPEG with HTTP 415 responses for invalid media types.
  - Added decompression bomb protection (`Image.MAX_IMAGE_PIXELS` capped at 25 MP) with HTTP 413.
  - Guarded against corrupt/truncated files, degenerate dimensions (<32x32), and extreme aspect ratios (>3.5:1) with HTTP 422.
  - Implemented heuristic OOD / low-confidence gate utilizing maximum softmax confidence (<0.50), Shannon predictive entropy (>1.30 nats), and image variance checks, returning `status="needs_review"` and `review_reason` without breaking backwards compatibility.

---

## [1.0.0] - 2026-10-12

### Documentation & Repository Audit
- Reconciled all documentation (`README.md`, `DEVELOPER_GUIDE.md`, `prd.md`) with verified codebase reality.
- Removed unverified claims regarding LIME and Grad-CAM++ (standard Grad-CAM verified; LIME and Grad-CAM++ confirmed absent from code).
- Published comprehensive [`MODEL_CARD.md`](MODEL_CARD.md) providing intended use, split protocols, verified metrics, and prominent clinical disclaimers.
- Published architectural overview in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and system verification matrix in [`docs/STATUS.md`](docs/STATUS.md).
- Added dedicated per-subsystem README files for `backend/`, `frontend/`, `ml/`, `dl/`, `infra/`, and `testing/`.
- Documented 106 passed tests (72% backend code coverage) and verified production frontend build.

---

## [0.9.0] - 2026-10-10

### Rigorous Evaluation & Leakage Mitigation
- **CT Leakage Resolution**: Discredited historical 99.63% accuracy claim after discovering 227 duplicate slice pairs across random shuffle splits. Replaced with 64-bit DCT perceptual hash grouping ($pHash \le 2$), establishing verified leak-free test accuracy of 86.94% (96.48% scan/cluster accuracy).
- **Tabular 50-Fold Nested CV**: Replaced legacy 12-sample test subset with 50-fold nested repeated stratified cross-validation, 1,000-sample bootstrap CIs, and permutation significance tests.
- **Model Selection (1-SE Rule)**: Evaluated Logistic Regression (ROC-AUC 0.8387), Random Forest (0.8175), and XGBoost (0.8076); deployed Logistic Regression as the parsimonious winner.
- **Deployment Gate Service**: Added automated gating criteria rejecting candidates with high regression, insufficient sensitivity, or leaky training partitions.

---

## [0.8.0] - 2026-09-25

### Security & Tenancy Hardening
- Implemented JWT access and refresh token authentication with bcrypt password hashing and sliding-window login rate limiting.
- Added role-based access control (`hospital_user`, `developer`, `admin`) with strict hospital query scoping via `get_scoped_hospital_id`.
- Added SHA-256 cryptographic audit logging table (`audit_logs`) tracking model deployment, rollback, and clinical authentication events.

---

## [0.7.0] - 2026-09-10

### Full-Stack Dashboard Overhaul & Persistence
- Integrated two dedicated consoles: Hospital Dashboard (stepped clinical case review) and Developer Dashboard (model management, telemetry, drift monitoring).
- Established relational persistence schema in `stonesense.db` (`hospitals`, `patients`, `predictions`, `federated_rounds`, `model_versions`, `system_logs`).
- Added real-time WebSocket connection manager (`ws_manager`) streaming live federated round lifecycle events to clients.

---

## [0.6.0] - 2026-08-25

### Federated Learning Extension
- Refactored centralized ResNet18 training into reusable `train_local` function.
- Implemented dataset partitioner supporting IID and Non-IID distributions across simulated hospitals (Apollo, Manipal, AIIMS).
- Built Flower (`flwr`) client (`StoneSenseFLClient`), gRPC server, and custom `StoneSenseFedAvg` aggregation strategy.
- Created multi-hospital simulation runner (`simulate.py`) and versioned checkpoint manager preserving `resnet18_fed_round_001.pth` through `029.pth`.

---

## [0.5.0] - 2026-08-15

### Multi-Model DL Wrappers & Registry
- Added unified deep learning registry in `dl/models/registry.py`.
- Implemented class wrappers for YOLO26 (`yolo26n_ct`), DINOv3 feature extractor, and QKNN quantum simulator.

---

## [0.4.0] - 2026-08-08

### Explainability Services & Clean Architecture
- Implemented dynamic SHAP explainer service in `ml/explainability/` switching between `LinearExplainer` and `TreeExplainer`.
- Implemented standard Grad-CAM hook service on ResNet18 `layer4`.
- Removed legacy `backend/main.py` stub in favor of modular `backend/app/main.py`.

---

## [0.1.0] - 2026-08-01

### Project Initialization
- Initialized repository with React 18 + Vite frontend and FastAPI backend stub.
- Initial exploratory data analysis on Andrews & Lyne urine biochemistry dataset.
