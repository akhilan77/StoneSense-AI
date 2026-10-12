# StoneSense-AI Testing Subsystem

The `testing/` directory contains automated test suites verifying model parity, data leakage prevention, feature integrity, deployment gate enforcement, federated learning simulation, and API stability.

---

## Test Inventory & Coverage Map

| Test Module | Description | Fast CI | Full Suite |
| :--- | :--- | :---: | :---: |
| `test_api_risk_regression.py` | Validates numerical stability of risk predictions across versions | Yes | Yes |
| `test_deployment_gate.py` | Comprehensive testing of deployment gate rules, rollback, and leaky checkpoint rejection | Yes | Yes |
| `test_dl_model_registry.py` | Unit tests for DL model wrapper instantiation, loading, and inference | Yes | Yes |
| `test_feature_source.py` | Asserts single authoritative feature column source (`feature_columns.pkl`) | Yes | Yes |
| `test_federated_api.py` | Tests REST endpoints for hospital federated learning status and local training | Yes | Yes |
| `test_grouped_split.py` | Asserts zero cross-split duplicate leakage on grouped CT partitions | Yes | Yes |
| `test_live_federated_round.py` | Live simulation of multi-hospital federated training and parameter aggregation | Marked `slow` | Yes |
| `test_local_training.py` | Tests isolated `train_local` execution and weight extraction | Yes | Yes |
| `test_ml_preprocessing_leakage.py` | Verifies data integrity, duplicate rejection, and scaling pipeline isolation | Yes | Yes |
| `test_model_registry.py` | Tests tabular `ModelRegistry` manifests, versions, and deployment state | Yes | Yes |
| `test_multi_model_inference.py` | Tests unified multi-model inference across ResNet18, YOLO26, DINOv3, and QKNN | Yes | Yes |
| `test_partition_leakage.py` | Cryptographic MD5 hash verification across all 12,446 CT image partitions | Skipped if archive unextracted | Yes |
| `test_partitioning.py` | Unit tests for IID and Non-IID dataset partitioning logic | Yes | Yes |
| `test_prediction_parity.py` | Tests that candidate models maintain output parity with reference baselines | Yes | Yes |
| `test_risk_explainability.py` | Asserts dynamic SHAP explainer selection (Linear vs Tree) and schema output | Yes | Yes |

---

## Running Test Suites

```bash
# Run all fast tests (recommended for pre-commit checks):
pytest -v -m "not slow"

# Run the complete test suite with code coverage:
pytest -v --cov=backend/app

# Run specific module:
pytest testing/test_deployment_gate.py -v
```

### Verified Test Run Results (October 2026 Audit)
- **Status:** **106 passed, 1 skipped, 0 failed** in 123.32s.
- **Coverage:** **72%** overall coverage across `backend/app` code.
