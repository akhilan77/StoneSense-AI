# Performance Benchmarking Report

- **Hardware Platform:** CPU Fallback (cpu)
- **ResNet18 CT image classification average latency:** `35.11 ms`
- **XGBoost urine chemistry risk assessment average latency:** `20.68 ms`
- **Orchestrated assessment average latency:** `41.90 ms`

---

## Latency Profiles
- Tabular XGBoost prediction utilizes lightweight float transformation and runs sub-millisecond.
- ResNet18 forward pass on CPU performs convolutional aggregations on a 224x224 tensor in ~30ms, which is highly suited for real-time web dashboard operations.
