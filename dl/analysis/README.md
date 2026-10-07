# DL Dataset & Cluster Evaluation Diagnostics

This directory contains standalone, reproducible diagnostic and validation scripts for the StoneSense-AI Deep Learning pipeline:

1. `cluster_error_audit.py`
   - Computes raw cluster-size distributions per split and per class (min, 25%, median, 75%, max, mean, std).
   - Audits multi-class clusters and inspects exact slice compositions.
   - Evaluates ResNet18 validation predictions per cluster and isolates the top 10 error clusters and error concentrations.

2. `frozen_features_grouped_cv.py`
   - Extracts 512-dimensional frozen backbone embeddings from ImageNet-pretrained ResNet18 across all 12,446 CT slices.
   - Runs a nested Stratified Group K-Fold Cross-Validation (`StratifiedGroupKFold`, 5 outer folds, 3 inner folds) grouped by `near_duplicate_group_id` (979 clusters).
   - Uses `StandardScaler` (fitted per training fold) and `LogisticRegression(class_weight='balanced')` with inner grid search across regularization parameter $C \in [0.001, 0.01, 0.1, 1.0, 10.0]$.
   - Reports unbiased out-of-fold generalization metrics (Accuracy, Macro F1, and per-class Recall $\pm$ Std).

### Usage
```bash
# Run cluster distribution and validation error audit
python dl/analysis/cluster_error_audit.py

# Run nested Stratified Grouped CV on frozen features
python dl/analysis/frozen_features_grouped_cv.py
```
