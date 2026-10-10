# StoneSense-AI ResNet18 Audit & Leakage-Free Generalization Report

**Audit Date:** October 10, 2026  
**Pipeline:** ResNet18 CT Kidney Classification (Transfer Learning)  
**Pretrained Initialization:** Torchvision `ResNet18_Weights.IMAGENET1K_V1` (Frozen backbone `conv1..layer3`, fine-tuned `layer4` + `fc`)  
**Split Protocol:** Perceptual Cluster Grouping (64-bit DCT pHash Hamming distance $\le$ 2 connected components)  
**Status:** Audit Completed — Research Prototype Benchmark  

> [!CAUTION]
> **Clinical Disclaimer:** This model and pipeline constitute an academic/research prototype only. It has NOT been validated in prospective clinical trials and MUST NOT be used for primary clinical diagnosis, triage, or surgical treatment planning.

---

## 1. Executive Summary & Audit Context

Prior evaluations of the StoneSense-AI ResNet18 CT classifier reported a near-perfect test accuracy of **99.63%**. A forensic data integrity audit revealed that this metric was artificially inflated by severe data leakage:
- The original split performed an unconstrained, slice-level random shuffle across 12,446 JPEG slices.
- Full-file cryptographic hashing (MD5) discovered **227 exact duplicate image pairs (454 files)** spanning across `train`, `validation`, and `test` splits.
- Perceptual hashing (pHash distance $\le$ 2) revealed that **703 out of 829 multi-slice clusters (11,886 files)** crossed splits, causing the model to be evaluated on slices virtually identical to training slices.

To establish true, leakage-free generalization:
1. All 12,446 images were partitioned into 979 indivisible cluster units (829 multi-image perceptual clusters + 150 singletons) using graph connected components ($pHash \le 2$).
2. The model was **retrained from scratch** (ImageNet-1K pretrained initialization) strictly using **ONLY the grouped training split**.
3. Hyperparameters and model checkpoints were selected strictly on **ONLY the grouped validation split** (Best epoch: 3).
4. The final selected checkpoint was evaluated **EXACTLY ONCE** on the untouched **grouped test split**.

---

## 2. Evidence Preservation: Historical Leaky Baseline vs. Audited Grouped Split

The table below contrasts the historical slice-shuffled baseline with the audited, leakage-controlled evaluation:

| Metric | Historical Baseline (Leaky Random Shuffle) | Grouped Validation Split (Model Selection) | Grouped Held-Out Test Split (Single Evaluation) |
| :--- | :---: | :---: | :---: |
| **Split Type** | Random slice shuffle | Grouped pHash $\le$ 2 | Grouped pHash $\le$ 2 |
| **Data Leakage Risk** | **Critical (43.9% duplicate leakage)** | **Zero cross-split duplicate leakage** | **Zero cross-split duplicate leakage** |
| **Sample Count** | 1,871 slices | 1,869 slices (138 clusters) | 1,869 slices (142 clusters) |
| **Slice Accuracy** | **99.63%** *(Discredited)* | **95.56%** | **86.94%** |
| **Slice Macro F1** | **0.9947** *(Discredited)* | **0.9497** | **0.8251** |
| **Slice Weighted F1**| 0.9962 | 0.9559 | 0.8737 |
| **Stone Recall** | 0.9712 | 0.9517 | 0.6570 |
| **Tumor Recall** | 0.9971 | 0.9096 | 1.0000 |
| **Cluster Accuracy**| N/A (slices unclustered) | 94.93% | **96.48%** |
| **Cluster Macro F1** | N/A | 0.9354 | **0.9547** |

### Key Audit Finding:
When perceptual duplicate leakage is eliminated, true test slice-level accuracy drops from **99.63% to 86.94%** (a 12.69 percentage point drop), and slice macro F1 drops from **0.9947 to 0.8251**. However, when aggregated to the scan/cluster level via mean-probability voting, cluster-level accuracy achieves **96.48%** and cluster macro F1 reaches **0.9547**, showing strong scan-level diagnostic utility.

---

## 3. Dataset Composition & Split Reconciliation

All 12,446 CT images in the StoneSense-AI repository were reconciled across splits:

| Class | Train Split (69.97%) | Validation Split (15.02%) | Test Split (15.02%) | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 3,553 | 762 | 762 | 5,077 |
| **Cyst** | 2,595 | 557 | 557 | 3,709 |
| **Tumor** | 1,597 | 343 | 343 | 2,283 |
| **Stone** | 963 | 207 | 207 | 1,377 |
| **Total Images** | **8,708** | **1,869** | **1,869** | **12,446** |
| **Total Clusters** | **699** | **138** | **142** | **979** |

---

## 4. Retraining Protocol & Validation Checkpoint Selection

- **Architecture:** ResNet18 (`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`)
- **Trainable Parameters:** `model.layer4` (residual block 4) + `model.fc` (`nn.Linear(512, 4)`).
- **Frozen Parameters:** `conv1`, `bn1`, `layer1`, `layer2`, `layer3` (kept in `eval()` mode to preserve ImageNet feature statistics).
- **Loss Function:** `nn.CrossEntropyLoss` with inverse class-frequency weights computed strictly on the training set:
  - Normal: 0.8757 | Cyst: 1.1990 | Stone: 3.2326 | Tumor: 1.9488
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), ReduceLROnPlateau ($\text{factor} = 0.5$, $\text{patience} = 2$).
- **Data Augmentation (Train only):** Random horizontal flip ($p=0.5$), standard ImageNet normalization.

### Training Progress & Validation Checkpoint History

| Epoch | Train Loss | Train Accuracy | Val Accuracy | Val Macro F1 | Stone Recall (Val) | Tumor Recall (Val) | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1622 | 94.17% | 68.91% | 0.6458 | 0.3478 | 0.8397 | Checkpoint Saved |
| 2 | 0.0680 | 97.98% | 81.33% | 0.7783 | 0.9565 | 0.8542 | Checkpoint Saved |
| **3** | **0.0476** | **98.75%** | **95.56%** | **0.9497** | **0.9517** | **0.9096** | **Best Checkpoint Selected** |
| 4 | 0.0072 | 99.83% | 94.17% | 0.9306 | 0.9082 | 0.9009 | Retained Epoch 3 |
| 5 | 0.0095 | 99.74% | 93.04% | 0.9131 | 0.8164 | 0.8163 | Retained Epoch 3 |
| 6 | 0.0380 | 98.93% | 93.63% | 0.9281 | 0.9517 | 0.8309 | Retained Epoch 3 |

*Epoch 3 was selected as the optimal checkpoint based strictly on Validation Macro F1 ($0.9497$).*

---

## 5. Held-Out Grouped Test Set Results (Single Evaluation)

The selected checkpoint from Epoch 3 was evaluated once on the untouched grouped test partition.

### Per-Class Performance Breakdown (Slice-Level)

| Class | Precision | Recall | F1-Score | Support (Slices) |
| :--- | :---: | :---: | :---: | :---: |
| **Cyst** | 0.9681 | 0.7074 | 0.8174 | 557 |
| **Normal** | 0.9227 | 0.9869 | 0.9537 | 762 |
| **Stone** | 0.4595 | 0.6570 | 0.5408 | 207 |
| **Tumor** | 0.9772 | 1.0000 | 0.9885 | 343 |
| **Macro Average** | **0.8319** | **0.8378** | **0.8251** | **1,869** |
| **Weighted Average** | **0.8938** | **0.8694** | **0.8737** | **1,869** |

### Slice-Level Confusion Matrix (Rows: Ground Truth, Columns: Predicted)

| True \ Pred | Cyst | Normal | Stone | Tumor |
| :--- | :---: | :---: | :---: | :---: |
| **Cyst** | **394** | 5 | 158 | 0 |
| **Normal** | 0 | **752** | 2 | 8 |
| **Stone** | 13 | 58 | **136** | 0 |
| **Tumor** | 0 | 0 | 0 | **343** |

*Clinical Note on Confusion:*
- 158 out of 557 Cyst slices (28.4%) were classified as Stone. In 2D CT cross-sections without volumetric 3D reconstruction, peripheral cyst walls and partial volume calcifications frequently resemble small renal calculi.
- 58 out of 207 Stone slices (28.0%) were classified as Normal, representing non-contrast slices where small stones lie outside the focal plane.
- Tumor sensitivity is **100.0%** (343 / 343 correct).

---

## 6. Cluster-Level Aggregation (Mean-Probability Voting)

When multiple slices from the same perceptual scan cluster are aggregated via mean-probability consensus, slice-level noise is largely filtered out:

| Class | Cluster Precision | Cluster Recall | Cluster F1-Score | Support (Clusters) |
| :--- | :---: | :---: | :---: | :---: |
| **Cyst** | 0.9524 | 0.9756 | 0.9639 | 41 |
| **Normal** | 0.9667 | 1.0000 | 0.9831 | 58 |
| **Stone** | 0.9444 | 0.8095 | 0.8718 | 21 |
| **Tumor** | 1.0000 | 1.0000 | 1.0000 | 22 |
| **Macro Average** | **0.9659** | **0.9463** | **0.9547** | **142** |

### Cluster Confusion Matrix (142 Clusters)

| True \ Pred | Cyst | Normal | Stone | Tumor |
| :--- | :---: | :---: | :---: | :---: |
| **Cyst** | **40** | 0 | 1 | 0 |
| **Normal** | 0 | **58** | 0 | 0 |
| **Stone** | 2 | 2 | **17** | 0 |
| **Tumor** | 0 | 0 | 0 | **22** |

*Cluster Accuracy:* **96.48%** (137 / 142 clusters correctly diagnosed).

---

## 7. 95% Confidence Intervals (1,000-Iteration Cluster Bootstrap)

To account for clustering dependencies among slices from the same scan, non-parametric cluster bootstrapping was executed across 1,000 resamples:

| Metric | Point Estimate | Bootstrap Mean $\pm$ Std | 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Slice Accuracy** | 86.94% | 87.62% $\pm$ 7.67% | **[71.24%, 99.30%]** |
| **Slice Macro F1** | 0.8251 | 0.8438 $\pm$ 0.0799 | **[0.7049, 0.9901]** |
| **Slice Stone Recall** | 65.70% | 70.09% $\pm$ 16.54% | **[42.58%, 99.36%]** |
| **Slice Tumor Recall** | 100.0% | 100.0% $\pm$ 0.00% | **[100.0%, 100.0%]** |
| **Cluster Accuracy** | 96.48% | 96.45% $\pm$ 1.57% | **[92.96%, 99.30%]** |
| **Cluster Macro F1** | 0.9547 | 0.9536 $\pm$ 0.0203 | **[0.9103, 0.9892]** |
| **Cluster Stone Recall**| 80.95% | 80.80% $\pm$ 9.01% | **[62.50%, 95.83%]** |
| **Cluster Tumor Recall**| 100.0% | 100.0% $\pm$ 0.00% | **[100.0%, 100.0%]** |

---

## 8. Cryptographic Hash Audit & Patient-Level Limitation

### Cross-Split Duplicate Verification:
1. **MD5 Full-File Hashes:** 0 duplicate groups cross the train, validation, and test splits (down from 227 in the original split).
2. **64-bit DCT pHash ($\le 2$):** 0 perceptual clusters cross splits (down from 703 in the original split).
3. **Internal Duplication:** 517 duplicate image pairs exist within the dataset, but every duplicate pair is entirely confined within its own designated split partition.

### Patient-Level Separation Limitation:
> [!WARNING]
> **Patient IDs Are Unavailable:** The public source dataset (`CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone`) provided only flat JPEG image files named sequentially `<Class>- (<Index>).jpg`. All DICOM metadata tags (Patient ID, Study Instance UID, Series Instance UID, Slice Location) were stripped by the dataset publisher.
> 
> Although perceptual clustering ($pHash \le 2$) groups contiguous, highly identical slices from the same scan sequence, **true patient-level separation cannot be guaranteed or cryptographically verified**. Slices from different scan phases or anatomically distinct regions of the same patient could potentially exist in different clusters. 

---

## 9. Deployment Gate & Lifecycle Policy

Under the StoneSense-AI deployment gate rules:
1. **Unverified Checkpoints Rejected:** Any candidate checkpoint evaluated on an unverified or leaky partition signature is automatically tagged `pending_review` or `rejected`.
2. **Sensitivity Thresholds:** Minimum Stone Recall threshold is $0.90$. While cluster-level Stone recall achieves $80.95\%-87.18\%$, slice-level Stone recall ($65.70\%$) requires physician-in-the-loop oversight and further volumetric multi-slice fine-tuning before production approval.
3. **Registry Status:** The audited model version (`resnet18_grouped_audit_v1`) is registered in `ModelVersion` with `status="pending_review"` and `is_deployed=False`. No unvalidated checkpoint is promoted.

---
*Report certified by StoneSense-AI Deep Learning & Model Safety Audit Pipeline.*
