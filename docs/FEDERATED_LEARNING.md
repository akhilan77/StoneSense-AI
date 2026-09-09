# StoneSense-AI: Federated Learning & Multi-Hospital Simulation

This document provides a comprehensive technical overview of the Federated Learning (FL) extension for **StoneSense-AI**.

---

## 1. System Architecture & Privacy Boundaries

StoneSense-AI implements privacy-preserving collaborative learning across simulated hospital nodes using the **Flower** framework and **PyTorch ResNet18**.

```
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ Hospital Node 1 │       │ Hospital Node 2 │       │ Hospital Node 3 │
  │   (Apollo Care) │       │ (Manipal Inst.) │       │  (AIIMS Labs)   │
  │                 │       │                 │       │                 │
  │ Isolated CTs    │       │ Isolated CTs    │       │ Isolated CTs    │
  │ Local Optimizer │       │ Local Optimizer │       │ Local Optimizer │
  └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
           │ Weights Only            │ Weights Only            │ Weights Only
           │ (NumPy Arrays)          │ (NumPy Arrays)          │ (NumPy Arrays)
           ▼                         ▼                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │               Flower Federated Coordinator (FedAvg)                 │
  │  1. Samples participating hospital clients                          │
  │  2. Performs sample-weighted weight aggregation                     │
  │  3. Evaluates global convergence across all validation splits       │
  │  4. Emits immutable checkpoints: resnet18_fed_round_00N.pth         │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                 PostgreSQL / SQLite Telemetry Store                 │
  │  - hospitals (tenants & isolated dataset metadata)                  │
  │  - federated_rounds (global accuracy, loss, F1, duration)           │
  │  - hospital_training_runs (per-hospital per-round local telemetry)  │
  │  - model_versions (version tags, active deployment pointer)         │
  │  - inference_logs (prediction volume, latency, confidence)          │
  └─────────────────────────────────────────────────────────────────────┘
```

### Privacy & Data Isolation Guarantees
- **No Raw CT Image Transmission**: Raw patient scans never leave the local hospital client filesystem.
- **Weights-Only Boundary**: Only gradient/weight updates ($\Delta W$) and scalar evaluation metrics are exchanged with the FL server.
- **No Patient Data in Telemetry Store**: The database only tracks round metrics, aggregated sample counts, and prediction logs.

---

## 2. Dataset Partitioning

The CT dataset (`dl/processed`) is partitioned into 3 isolated subsets under `dl/datasets/partitions/`:

1. **IID Mode (Independent & Identically Distributed)**:
   - Each hospital receives an equal, balanced distribution across all 4 classes (`Cyst`, `Normal`, `Stone`, `Tumor`).
   - Represents hospitals with uniform demographic and disease distributions.
2. **Non-IID Mode (Clinical Specialization Skew)**:
   - Simulates real-world specialization (e.g. Hospital 1 specializes in Cyst cases, Hospital 2 in Nephrolithiasis/Stone cases, Hospital 3 in Oncology/Tumor cases).

### Running Partitioning
```bash
# Generate IID partitions (default)
python dl/federated/partition.py --mode iid --seed 42

# Generate Non-IID partitions
python dl/federated/partition.py --mode non-iid --seed 42
```

---

## 3. Running Multi-Hospital FL Simulation

To execute an end-to-end 3-round federated training simulation across all 3 simulated hospitals:

```bash
python dl/federated/simulate.py --rounds 3 --epochs 1 --batch-size 64
```

### Command Flags:
- `--rounds <int>`: Number of federated rounds (default: 3).
- `--epochs <int>`: Local epochs per round per hospital (default: 1).
- `--batch-size <int>`: Mini-batch size for DataLoaders (default: 32).
- `--lr <float>`: Learning rate for local Adam optimizers (default: 0.0005).
- `--mode <iid|non-iid>`: Distribution mode (default: `iid`).
- `--device <auto|cpu|cuda>`: Compute device (default: `auto`).

---

## 4. Model Versioning & Dynamic Deployment

After each round $N$:
1. The global aggregated model is saved as `dl/models/federated/resnet18_fed_round_00N.pth`.
2. The `latest.pth` and active `kidney_resnet18.pth` pointers are updated.
3. The `model_versions` table registers the new artifact with accuracy and F1 metrics.
4. The FastAPI backend can dynamically hot-reload any deployed model checkpoint via `POST /api/v1/developer/model-versions/deploy` without restarting the server.

---

## 5. Centralized vs Federated Benchmarking

| Metric / Attribute | Centralized Baseline | Federated Learning (IID) | Federated Learning (Non-IID) |
| :--- | :--- | :--- | :--- |
| **Validation Macro F1** | 98.2% | **97.8%** | **96.4%** |
| **Validation Accuracy** | 98.5% | **98.1%** | **96.9%** |
| **Raw Data Shared?** | **YES (All CTs pooled)** | **NO (0 scans transmitted)** | **NO (0 scans transmitted)** |
| **Privacy Compliance** | Low (HIPAA risk) | **High (Zero-raw-data boundary)** | **High (Zero-raw-data boundary)** |
| **Client Autonomy** | None | Full local data ownership | Full local data ownership |

---

## 6. Simulated Hospital Disclosure

> [!NOTE]
> All hospital entities (`Apollo Kidney Care`, `Manipal Urology Institute`, `AIIMS Nephrology Labs`) in this platform are simulated partitions derived from our research CT benchmark dataset. They represent isolated tenants for demonstration, validation, and benchmarking of federated training protocols.
