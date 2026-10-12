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
