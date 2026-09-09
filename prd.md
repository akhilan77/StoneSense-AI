**PRODUCT REQUIREMENTS DOCUMENT — StoneSense-AI Federated Learning Extension**

---

**1. DOCUMENT INFO**
1. Project: StoneSense-AI — Federated Learning + Multi-Hospital Simulation Extension
2. Version: v1.0 (Draft)
3. Owner: akhilan77
4. Related repo: StoneSense-AI (existing ML/DL/XAI platform)

---

**2. PROBLEM STATEMENT**
1. StoneSense-AI currently trains its DL (ResNet18) model on a single centralized dataset.
2. Real-world hospital data cannot be centralized due to privacy constraints.
3. There is no mechanism to simulate, coordinate, or evaluate multi-hospital collaborative training.
4. There is no persistent record of training history, model versions, or hospital participation.
5. Developer and Hospital pages currently have no live/statistical backing — they are static or disconnected from any real training process.

---

**3. GOAL**
1. Add a federated learning layer around the existing ResNet18 pipeline so multiple simulated hospitals can collaboratively train a shared model without sharing raw CT data.
2. Persist all training/round/model metadata so it can be analyzed and displayed.
3. Expose this system through the Hospital Page (client view) and Developer Page (server/admin view).

---

**4. NON-GOALS**
1. Not replacing XGBoost/SHAP (ML module stays centralized).
2. Not replacing ResNet18 architecture or Grad-CAM.
3. Not implementing differential privacy, secure aggregation, blockchain, or advanced FL algorithms (FedProx, SCAFFOLD) in this phase.
4. Not deploying to real multi-machine/hospital infrastructure — local simulation only.
5. Not storing raw CT images in PostgreSQL or transmitting them to the FL server.

---

## PRD PART A — CORE FUNCTIONALITY CHANGES
(These define whether federated learning actually works. Without these, nothing else has real data to display.)

**A1. Local Training Refactor**
1. Requirement: Convert existing centralized `train_resnet18.py` logic into a reusable function `train_local(model, dataloader, config) → model, metrics`.
2. Acceptance criteria: Function can be invoked independently per hospital dataset; returns model weights + accuracy/precision/recall/F1/loss/sample count.
3. Priority: P0 (blocking everything else).

**A2. Dataset Partitioning**
1. Requirement: Utility to split the existing CT dataset into 3 simulated hospital partitions.
2. Must support both IID (similar class distribution) and non-IID (skewed class distribution) modes, seeded for reproducibility.
3. Acceptance criteria: Given a seed and mode, produces consistent hospital_1/2/3 train/val/test splits with logged class distributions.
4. Priority: P0.

**A3. Flower FL Client**
1. Requirement: Implement `NumPyClient` (or current Flower `ClientApp`) per hospital with `get_parameters`, `set_parameters`, `fit`, `evaluate`.
2. Acceptance criteria: Each client trains only on its own partition, returns weights + sample count + metrics, never transmits raw images.
3. Priority: P0.

**A4. Flower FL Server + FedAvg**
1. Requirement: Implement `ServerApp`/`start_server` with `FedAvg` strategy (min_fit_clients=3, min_evaluate_clients=3, configurable rounds/epochs/lr/batch size).
2. Acceptance criteria: Running the server + 3 clients locally completes N rounds, produces a converging global model, with round-by-round metrics printed/logged.
3. Priority: P0.

**A5. Model Versioning / Model Manager**
1. Requirement: After each round, save global model as a new versioned checkpoint (`resnet18_fed_round_00N.pth`) and update a `latest.pth` pointer. Never overwrite prior versions.
2. Acceptance criteria: N completed rounds → N distinct checkpoint files retrievable by round number.
3. Priority: P0.

**A6. Database Persistence Layer (Schema + Connection)**
1. Requirement: PostgreSQL + SQLAlchemy setup with 5 tables: `hospitals`, `federated_rounds`, `hospital_training_runs`, `model_versions`, `inference_logs`.
2. Acceptance criteria: Backend connects via `DATABASE_URL` env var; tables created via migration; no hardcoded credentials; no raw image data stored.
3. Priority: P0 (required to persist A3-A5 outputs — without it results are lost when the process ends).

**A7. Hospital ↔ Global Model Synchronization**
1. Requirement: After a round completes, each hospital's "current model version" updates to the new global model, and this becomes the model used for live CT inference.
2. Acceptance criteria: Post-round, hospital inference endpoint serves predictions using the newly aggregated model, verified by version-tag in response.
3. Priority: P0.

**A8. Local Simulation Runnable End-to-End**
1. Requirement: A documented command sequence to run FL server + 3 hospital clients locally and complete at least 3 full rounds without manual intervention.
2. Acceptance criteria: Round 1 → Round 3 completes with logged accuracy improving or stabilizing; no raw data leaves each client's process boundary.
3. Priority: P0 — this is the acceptance gate before any UI work begins.

---

## PRD PART B — TOUCH-UP / SUPPORTING CHANGES
(These expose and record what Part A produces. System is "functional" without these but not usable, demonstrable, or auditable.)

**B1. Backend REST APIs (exposure layer)**
1. Hospital APIs: list hospitals, hospital detail, training-history, current model, federated-status (5 endpoints).
2. Developer APIs: federated overview, round history, hospital participation, model versions, statistics, system-stats (6 endpoints).
3. Local-data workflow APIs: dataset status, dataset validate, start local training, training status (4 endpoints).
4. Priority: P1 — required for UI, not for FL correctness.

**B2. Statistical/Analytics Tables**
1. `hospital_training_runs` — per-hospital per-round metrics logging.
2. `inference_logs` — prediction count, latency, confidence tracking.
3. Priority: P1 — enhances Developer Page insights; not required for FedAvg to function.

**B3. Hospital Page UI**
1. Display: hospital status, current global model version, current round, local dataset size/class distribution, last training result, local accuracy/F1.
2. Actions: "Validate Local Dataset," "Start Local Training," "Refresh Model Status."
3. Existing CT upload → inference → Grad-CAM flow preserved unchanged.
4. Priority: P1.

**B4. Developer Page UI**
1. Overview cards: active hospitals, current round, global model version, global accuracy/F1, total samples/rounds.
2. Charts: accuracy/F1/loss/participation vs round, per-hospital sample contribution, per-hospital local performance comparison.
3. Tables: round history, hospital participation, model version history.
4. Priority: P1.

**B5. Frontend API Service Layer**
1. Typed Axios functions + TypeScript interfaces for all B1 endpoints.
2. Loading/error/empty states for every new data-fetching component.
3. Priority: P1.

**B6. Documentation**
1. Architecture diagram, setup instructions (PostgreSQL, backend, frontend, FL server, FL clients), experiment methodology, explicit disclosure that hospitals are simulated partitions (not real institutions).
2. Priority: P2.

**B7. Research Extensions (Optional/Stretch)**
1. Non-IID vs IID comparative experiment and writeup.
2. Centralized vs Federated performance comparison table (accuracy/precision/recall/F1/communication overhead/raw-data-sharing requirement).
3. Priority: P2 — valuable for thesis/report, not required for MVP demo.

---

**5. SEQUENCING / DEPENDENCY RULE**
1. Part A (A1→A8) must be completed and verified via command-line simulation before any Part B work starts.
2. Rationale: Building B3/B4 dashboards before A6/A7 exist means displaying fake or empty data — wastes effort and risks misrepresenting results.
3. Recommended order: A1 → A2 → A3+A4 → A5 → A6 → A7 → A8 (gate check) → B1 → B2 → B3+B4 → B5 → B6 → B7.

---

**6. SUCCESS METRICS**
1. FL loop completes ≥3 rounds locally with 3 simulated clients, no raw CT data crossing client boundary (verified by code inspection/logging).
2. Global model accuracy trend is logged and non-fabricated across rounds.
3. PostgreSQL contains accurate round/model/hospital records matching simulation logs.
4. Developer Page charts match values in the database (spot-check ≥3 rounds).
5. Hospital Page correctly reflects current global model version after each round.
6. Existing ML (XGBoost/SHAP) and Grad-CAM functionality unaffected (regression check).

---

**7. RISKS**
1. Risk: Copilot may implement "fake FL" (train centrally, average after) — mitigate by requiring true per-client local training with isolated data loaders.
2. Risk: Outdated Flower API usage from old tutorials — mitigate by checking installed Flower version's current ServerApp/ClientApp docs before implementation.
3. Risk: Claiming real multi-hospital data when it's simulated — mitigate via explicit documentation (B6).
4. Risk: PostgreSQL misuse to store raw images — mitigate via schema review (no BLOB/image fields in any table).