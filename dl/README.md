# StoneSense-AI Deep Learning Subsystem (Axial CT Vision & Federated Learning)

The `dl/` directory contains convolutional classification models for axial CT kidney scans, perceptual cluster split logic to eliminate slice leakage, Grad-CAM visual heatmaps, and decentralized federated learning coordination using the Flower (`flwr`) framework.

---

## Directory Layout

```text
dl/
├── explainability/
│   └── gradcam.py            # Standard Grad-CAM implementation (layer4[-1])
├── federated/                # Multi-hospital Federated Learning (Flower)
│   ├── client.py             # Flower NumPyClient per simulated hospital
│   ├── local_training.py     # Isolated local training function: train_local()
│   ├── model_manager.py      # Versioned checkpoint saving & latest pointer
│   ├── partition.py          # IID and Non-IID dataset partitioner
│   ├── server.py             # Flower gRPC server launcher
│   ├── simulate.py           # Multi-hospital end-to-end simulation runner
│   └── strategy.py           # StoneSenseFedAvg custom FedAvg strategy
├── models/
│   ├── ct/                   # Architecture checkpoints & metrics
│   │   ├── resnet18/         # ResNet18 checkpoints & audited metrics.json
│   │   ├── yolo26/           # YOLO26 classifier weights (best.pt) & predictions.csv
│   │   ├── dinov3/           # DINOv3 feature extractor backbone & predictions.csv
│   │   └── qknn/             # Quantum KNN prototype & predictions.csv
│   ├── federated/            # Versioned FL checkpoints (resnet18_fed_round_001.pth - 029.pth)
│   └── wrappers/             # Modular inference wrappers (ResNet18, YOLO26, DINOv3, QKNN)
├── preprocessing/            # Image loaders, transforms, and perceptual hash grouping
│   ├── dataloaders.py        # PyTorch DataLoader factories
│   ├── grouping.py           # 64-bit DCT pHash clustering connected components
│   └── transforms.py         # Standard Torchvision normalization & augmentation
└── training/
    ├── evaluate.py           # Multiclass evaluation (Confusion Matrix, Recall, F1)
    ├── model.py              # build_resnet18_classifier() architecture factory
    ├── train_resnet18.py     # Centralized ResNet18 training pipeline
    └── train_resnet18_audit.py # Audited leakage-free grouped training pipeline
```

---

## Data Leakage Audit & Verified Metrics

Prior evaluations reported a near-perfect accuracy of **99.63%** on axial CT slices. An engineering audit discovered:
- The historical random shuffle split suffered from **227 exact cross-split duplicate image pairs** and **703 perceptual clusters crossing train and test splits**.
- Slices were reorganized into 979 indivisible cluster units using 64-bit DCT perceptual hashing ($pHash \le 2$).
- ResNet18 was retrained exclusively on the grouped training set and evaluated on the held-out grouped test set (1,869 slices, 142 clusters).

The restored Kaggle release was independently regrouped by `dl/preprocessing/generate_grouped_split.py`: 12,446 images produced 979 connected components (829 multi-image components), including 517 SHA-256-identical pairs (1,034 images in those pairs) and 6,953 unique pHash edges at distance <=2. This global exact-duplicate pair count is not directly comparable to the historical count of pairs that crossed the old split. The deterministic grouped allocation is train 8,712, validation 1,867, test 1,867; class counts are in `dl/outputs/reports/phash_clusters.json`. All classes occur in all three splits and no group crosses partitions. This demonstrates image-level duplicate isolation only; patient IDs are unavailable.

### Audited Benchmark Results (Source: [`dl/models/ct/resnet18/metrics.json`](models/ct/resnet18/metrics.json) & [`docs/RESNET18_AUDIT_REPORT.md`](../docs/RESNET18_AUDIT_REPORT.md))

| Metric | Audited Leak-Free Held-Out Test Set | Historical Leaky Baseline (Discredited) |
| :--- | :---: | :---: |
| **Slice Accuracy** | **86.94%** (1,625 / 1,869) | 99.63% |
| **Slice Macro F1** | **82.51%** | 99.47% |
| **Cyst Recall** | **70.74%** (394 / 557) | 99.82% |
| **Normal Recall** | **98.69%** (752 / 762) | 100.0% |
| **Stone Recall (Sensitivity)** | **65.70%** (136 / 207) | 97.12% |
| **Tumor Recall (Sensitivity)** | **100.0%** (343 / 343) | 99.71% |
| **Scan / Cluster Accuracy (Majority Vote)** | **96.48%** (137 / 142) | N/A |
| **Scan / Cluster Macro F1** | **95.47%** | N/A |

---

## QKNN Pipeline and Validation

QKNN is a quantum-kernel nearest-neighbor classifier over frozen 384-dimensional DINOv3 CLS embeddings. Training applies a train-fitted scaler/PCA representation, selects balanced training prototypes, and scores them using a PennyLane `default.qubit` fidelity circuit with Y-axis angle encoding. It is a state-vector simulation, not a quantum-hardware model. The current circuit has no entangling gates; its fidelity can therefore be evaluated exactly as the product of per-coordinate `cos²((x - prototype) / 2)` terms.

`dl/models/wrappers/qknn_wrapper.py` supports both the current `qknn_bundle.joblib` export (including its validation-selected `k`) and the older `qknn_model.pkl`/`pca_pipeline.pkl`/`prototypes.npz` artifacts. Transform order is resolved from the fitted scaler/PCA input dimensions so inference follows the saved training pipeline. Run `python dl/analysis/evaluate_qknn_validation.py` to repeat the neighbor-count and voting-weight sweep on validation embeddings; it uses validation only for selection, then reports one final test evaluation.

The checked-in legacy QKNN test report is 41.95% accuracy and 35.89% macro F1. On the available validation embeddings, similarity-weighted `k=9` improves macro F1 from 40.82% (`k=40`) to 42.78%; that same setting did not improve the held-out test metrics, so the legacy model's default behavior is intentionally unchanged. Test results must be read from the existing prediction export and are not a tuning target.

### Grouped-data prerequisites for QKNN notebooks

`dl/notebooks/03_qknn.ipynb` accepts embedding archives with either `X`/`y`/`paths` or `embeddings`/`labels`/`paths` (the extraction notebook also retains `filenames`). Labels are integer indices `Cyst=0`, `Normal=1`, `Stone=2`, `Tumor=3`. It requires the authoritative `dl/outputs/reports/grouped_split_manifest.csv` with `filename`, `class`, `destination_split`, and `near_duplicate_group_id`; cached group IDs, when present, are checked against it. The notebook rejects incomplete split coverage, path/label mismatches, and any group appearing in multiple splits. It does not infer groups from filenames or path uniqueness.

`dl/notebooks/01_dinov3.ipynb` writes the real manifest group IDs and paths into each embedding cache. An incomplete existing cache is not silently reused; set `RERUN=True` only after restoring the grouped images and manifest. Before replacing any existing cache, the notebook makes a timestamped backup. QKNN experiment exports go to `dl/models/ct/qknn_candidates/qknn/` so they cannot overwrite the legacy baseline.

### Rebuilding the image-level duplicate-aware split

The restored Kaggle images can be grouped without modifying the source tree:

```powershell
python dl\preprocessing\generate_grouped_split.py `
  --source-dir "D:\Data\StoneSenseAI\datasets\ct-kidney-kaggle\extracted\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
```

The command hashes every source image with SHA-256, computes a 64-bit DCT pHash, and connects images whose pHash Hamming distance is at most 2. Exact duplicate hashes are also joined. Connected components are indivisible groups; a deterministic, class-stratified greedy allocator targets 70/15/15 and preserves every class in every split. It copies images to the ignored `dl/processed_grouped/` tree and writes `grouped_split_manifest.csv`, `duplicate_analysis.csv`, `phash_clusters.json`, and `phash_duplicates.csv` beneath ignored `dl/outputs/reports/`. It refuses to overwrite any of those outputs; inspect or move them deliberately before any future regeneration.

Connected components are transitive: two images in the same component may be more than two pHash bits apart if linked by intermediate images. pHash is a visual near-duplicate heuristic and can both merge distinct images and miss related images; SHA-256 only identifies byte-identical files. The Kaggle release has no patient identifiers, so these outputs establish image-level exact/near-duplicate isolation only—not patient-level separation.

After a new split is generated, old embedding caches must not be treated as matching it unless their cached paths have the same assigned split. The existing caches currently contain records assigned to different train/validation/test partitions than the newly generated manifest (4,888 of 12,446 cached records disagree on split). The QKNN notebook's manifest join should reject them. Do not evaluate or tune against those caches as though they were generated from this split. Re-extracting embeddings is a separate, explicit follow-up.

---

## Explainability: Standard Grad-CAM

Visual heatmaps are produced using **standard Grad-CAM** (Selvaraju et al., 2017) targeting the convolutional feature maps of `layer4[-1]`:
- Gradients of the predicted class logit are pooled to generate channel weights.
- The resulting weighted activation map is normalized and blended with the input image.
- *Audit Finding:* The repository does **not** implement Grad-CAM++ or LIME.

---

## Federated Learning Workflow

Decentralized multi-hospital training simulates 3 clinical sites (Apollo Kidney Care, Manipal Urology Institute, AIIMS Nephrology Labs) using Flower (`flwr`):
1. **Partitioning:** `dl/federated/partition.py` allocates image clusters to hospital partitions.
2. **Local Training:** `train_local()` trains ResNet18 independently per hospital for $E$ local epochs.
3. **Weight Aggregation:** `StoneSenseFedAvg` computes sample-weighted averages of model weights.
4. **Checkpoint Preservation:** Saves versioned files (`resnet18_fed_round_00N.pth`) to `dl/models/federated/`.
5. **Telemetry Persistence:** Telemetry is stored directly in `stonesense.db`.

```bash
# Run local 3-round simulation:
python dl/federated/simulate.py --rounds 3 --epochs 1 --mode iid
```
