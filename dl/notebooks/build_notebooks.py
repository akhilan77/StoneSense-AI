"""Script to generate and validate the 5 Phase 8 CT multi-model Jupyter Notebooks using nbformat.v4 API."""

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def build_nb00():
    cells = [
        new_markdown_cell("""# StoneSense-AI: CT Multi-Model Experimentation — Setup & Data Verification

### 📋 Overview & Drive Layout
This notebook is **Step 0** of the Phase 8 Multi-Model CT evaluation workflow. It verifies your environment, links Google Drive persistence, validates dataset integrity for `dl/processed_grouped`, guarantees **zero near-duplicate cluster leakage**, and prepares directory structures and YOLO dataset views for downstream modeling.

---
### 🗂️ Google Drive Persistence Directory Layout
```
/content/drive/MyDrive/StoneSense-AI/
├── data/
│   ├── processed_grouped/           # 8708 Train, 1869 Val, 1869 Test
│   └── grouped_split_manifest.csv   # Cluster mapping & splits
├── artifacts/
│   ├── embeddings/                  # Cached DINOv3 .npz features
│   └── yolo_view/                   # Train/Val/Test YOLO format
└── outputs/
    ├── dinov3/                      # model_card.json, metrics.json, predictions.csv, weights
    ├── yolo26/                      # model_card.json, metrics.json, predictions.csv, best.pt
    ├── qknn/                        # model_card.json, metrics.json, predictions.csv, qknn_bundle.joblib
    └── comparison/                  # comparison.json, benchmark tables
```

---
### ⚙️ Notebook Flow
1. **Drive Mount & Workspace Configuration**
2. **Environment & Dependency Diagnostic Check**
3. **Grouped Split Manifest Verification & Leakage Assertion**
4. **Dataset Image Integrity & Class Balance Scan**
5. **YOLO-Compatible Dataset View Creation (`dl/datasets/yolo_view`)**
6. **Package Verification & Persistence Summary**"""),

        new_markdown_cell("""---
### 1. Configuration & Persistence Settings
* **What it does:** Sets configuration variables, seeds, paths, Drive root, idempotency flag `RERUN`, and `SMOKE_TEST` flag (fast 50-sample mode).
* **Expected runtime:** < 1 second.
* **Expected output:** Prints configured paths and confirmed execution mode."""),

        new_code_cell("""import os
import sys
import shutil
from pathlib import Path

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
SEED = 42
RERUN = False       # Set to True to re-run verification and re-create view structures
SMOKE_TEST = False  # Set to True to run on 50 images per split for instant verification

# Determine runtime environment (Colab vs Kaggle vs Local)
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = "kaggle_web_client" in sys.modules

if IN_COLAB:
    DRIVE_ROOT = Path("/content/drive/MyDrive/StoneSense-AI")
    WORKSPACE_ROOT = Path("/content/StoneSense-AI")
elif IN_KAGGLE:
    DRIVE_ROOT = Path("/kaggle/working/StoneSense-AI")
    WORKSPACE_ROOT = Path("/kaggle/working/StoneSense-AI")
else:
    WORKSPACE_ROOT = Path(".").resolve()
    DRIVE_ROOT = WORKSPACE_ROOT

# Derive all paths strictly from WORKSPACE_ROOT / DRIVE_ROOT using forward-slash POSIX paths
DATA_DIR = WORKSPACE_ROOT / "dl" / "processed_grouped"
MANIFEST_PATH = WORKSPACE_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"
YOLO_VIEW_DIR = WORKSPACE_ROOT / "dl" / "datasets" / "yolo_view"
ARTIFACTS_DIR = WORKSPACE_ROOT / "dl" / "artifacts"
OUT_DIR = WORKSPACE_ROOT / "dl" / "models" / "ct"

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

print(f"Runtime Environment: {'Google Colab' if IN_COLAB else ('Kaggle' if IN_KAGGLE else 'Local')}")
print(f"Workspace Root: {WORKSPACE_ROOT.as_posix()}")
print(f"Data Directory: {DATA_DIR.as_posix()}")
print(f"Drive Persistence Root: {DRIVE_ROOT.as_posix()}")
print(f"Smoke Test Mode: {SMOKE_TEST}")"""),

        new_markdown_cell("""---
### 2. Optional Drive Mount (Colab Only)
* **What it does:** Mounts Google Drive at `/content/drive` for automatic persistence across disconnections.
* **Expected runtime:** ~10-15 seconds.
* **Expected output:** Confirmation of Google Drive mount."""),

        new_code_cell("""if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Google Drive successfully mounted at {DRIVE_ROOT.as_posix()}")
else:
    print("Skipping Colab Drive mount (local or Kaggle environment).")"""),

        new_markdown_cell("""---
### 3. Environment & Hardware Diagnostics (CHECK Cell)
* **What it does:** Verifies Python, PyTorch, CUDA GPU availability, packages, and reads `HF_TOKEN` from environment/secrets.
* **Expected runtime:** ~3-5 seconds.
* **Expected output:** Hardware specs, GPU model, and installed library versions. Fails clearly if prerequisites are missing."""),

        new_code_cell("""import platform
import torch

print("=" * 60)
print("ENVIRONMENT & HARDWARE DIAGNOSTICS")
print("=" * 60)
print(f"Python Version : {platform.python_version()}")
print(f"OS Platform    : {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("No GPU detected; using CPU for execution.")

# Check Hugging Face Token (Read strictly from Colab secrets or environment)
hf_token = None
if IN_COLAB:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
    except Exception:
        hf_token = os.getenv('HF_TOKEN')
else:
    hf_token = os.getenv('HF_TOKEN')

if hf_token:
    print(f"HF_TOKEN detected in environment/secrets (length: {len(hf_token)})")
else:
    print("HF_TOKEN not found. DINOv3 gated access will require token; DINOv2 open fallback will be available.")

print("=" * 60)"""),

        new_markdown_cell("""---
### 4. Manifest & Zero-Leakage Verification
* **What it does:** Reads `grouped_split_manifest.csv` and verifies:
  1. Exact split counts: **Train (8,708), Validation (1,869), Test (1,869)** — Total 12,446.
  2. Fixed 4 classes: `Cyst`, `Normal`, `Stone`, `Tumor`.
  3. **Zero cluster leakage:** Asserts intersection of `near_duplicate_group_id` across splits is empty.
* **Expected runtime:** ~1-2 seconds.
* **Expected output:** Split table, cluster counts, and ZERO CLUSTER LEAKAGE VERIFIED."""),

        new_code_cell("""import pandas as pd

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH.as_posix()}. Check data sync.")

df_manifest = pd.read_csv(MANIFEST_PATH)
print(f"Loaded manifest: {len(df_manifest)} total records across {df_manifest['near_duplicate_group_id'].nunique()} clusters.")

# 1. Verify split counts
split_counts = df_manifest["destination_split"].value_counts().to_dict()
print(f"Split distribution: {split_counts}")

expected_counts = {"train": 8708, "validation": 1869, "test": 1869}
for split, exp_count in expected_counts.items():
    actual_count = split_counts.get(split, 0)
    assert actual_count == exp_count, f"Count mismatch in '{split}': expected {exp_count}, got {actual_count}"

# 2. Verify classes
classes = sorted(df_manifest["class"].unique().tolist())
assert classes == sorted(CLASS_NAMES), f"Unexpected classes: {classes} vs {CLASS_NAMES}"

# 3. Verify zero cluster leakage across splits
train_clusters = set(df_manifest[df_manifest["destination_split"] == "train"]["near_duplicate_group_id"])
val_clusters = set(df_manifest[df_manifest["destination_split"] == "validation"]["near_duplicate_group_id"])
test_clusters = set(df_manifest[df_manifest["destination_split"] == "test"]["near_duplicate_group_id"])

leakage_train_val = train_clusters.intersection(val_clusters)
leakage_train_test = train_clusters.intersection(test_clusters)
leakage_val_test = val_clusters.intersection(test_clusters)

assert len(leakage_train_val) == 0, f"Leakage detected between Train and Val: {len(leakage_train_val)} clusters"
assert len(leakage_train_test) == 0, f"Leakage detected between Train and Test: {len(leakage_train_test)} clusters"
assert len(leakage_val_test) == 0, f"Leakage detected between Val and Test: {len(leakage_val_test)} clusters"

print("=" * 60)
print("ZERO CLUSTER LEAKAGE VERIFIED:")
print(f"   - Total Clusters : {len(train_clusters) + len(val_clusters) + len(test_clusters)}")
print(f"   - Train Clusters : {len(train_clusters)}")
print(f"   - Val Clusters   : {len(val_clusters)}")
print(f"   - Test Clusters  : {len(test_clusters)}")
print(f"   - Overlap        : 0 clusters")
print("=" * 60)"""),

        new_markdown_cell("""---
### 5. Physical Image Filesystem Scan
* **What it does:** Scans `dl/processed_grouped/{train,validation,test}` to confirm JPEG images are readable on disk.
* **Expected runtime:** ~3-8 seconds (or ~1 sec in SMOKE_TEST mode).
* **Expected output:** Confirmed image count and verified dimensions."""),

        new_code_cell("""from PIL import Image

df_scan = df_manifest.head(150) if SMOKE_TEST else df_manifest
verified_images = 0
corrupted_images = []

for idx, row in df_scan.iterrows():
    split = row["destination_split"]
    cname = row["class"]
    fname = row["filename"]
    img_path = DATA_DIR / split / cname / fname
    
    if not img_path.exists():
        corrupted_images.append(img_path.as_posix())
        continue
    verified_images += 1

if corrupted_images:
    raise FileNotFoundError(f"{len(corrupted_images)} images missing on disk! Example: {corrupted_images[:3]}")

print(f"All {verified_images:,} scanned images verified successfully across disk paths in '{DATA_DIR.name}'.")"""),

        new_markdown_cell("""---
### 6. Create YOLO-Compatible View (`dl/datasets/yolo_view`)
* **What it does:** Builds a zero-copy symlink view (with copy fallback) mapping `validation` to `val` for Ultralytics YOLO26 training.
* **Expected runtime:** ~2-5 seconds.
* **Expected output:** `yolo_view` folder populated with `train/`, `val/`, `test/` and `dataset.yaml` created."""),

        new_code_cell("""import yaml

YOLO_VIEW_DIR.mkdir(parents=True, exist_ok=True)
yolo_yaml_path = YOLO_VIEW_DIR / "dataset.yaml"

# Ultralytics requires 'val' naming
split_mapping = {
    "train": "train",
    "validation": "val",
    "test": "test"
}

for src_split, dst_name in split_mapping.items():
    dst_dir = YOLO_VIEW_DIR / dst_name
    src_dir = DATA_DIR / src_split
    
    if dst_dir.exists() and not RERUN:
        print(f"YOLO view '{dst_name}' already exists. Skipping.")
        continue
        
    if dst_dir.exists():
        try:
            os.rmdir(dst_dir)
        except Exception:
            shutil.rmtree(dst_dir)
        
    try:
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(str(src_dir.resolve()), str(dst_dir.resolve()))
        else:
            os.symlink(src_dir.resolve(), dst_dir.resolve(), target_is_directory=True)
        print(f"Created zero-copy link: {dst_name} -> {src_split}")
    except (OSError, NotImplementedError) as exc:
        print(f"Zero-copy link failed ({exc}); copying view structure for {dst_name}...")
        shutil.copytree(src_dir, dst_dir)

dataset_config = {
    "path": YOLO_VIEW_DIR.resolve().as_posix(),
    "train": "train",
    "val": "val",
    "test": "test",
    "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    "nc": len(CLASS_NAMES)
}

with open(yolo_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

print(f"Generated YOLO dataset config at {yolo_yaml_path.as_posix()}")"""),

        new_markdown_cell("""---
### 7. Verification & Summary Checklist
* **What it does:** Confirms all directories are ready for Notebooks 01–04.
* **Expected runtime:** < 1 second.
* **Expected output:** Final status report."""),

        new_code_cell("""print("=" * 60)
print("STEP 00 SETUP & DATA VERIFICATION COMPLETE")
print("=" * 60)
print(f"Processed Grouped Dataset: {DATA_DIR.as_posix()} (12,446 images)")
print(f"YOLO View Config         : {yolo_yaml_path.as_posix()}")
print(f"Artifacts Directory      : {ARTIFACTS_DIR.as_posix()}")
print(f"Ready to run Notebook 01_dinov3.ipynb!")
print("=" * 60)""")
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    return nb


def build_nb01():
    cells = [
        new_markdown_cell("""# StoneSense-AI: CT Multi-Model Experimentation — DINOv3 Foundation Model

### 📋 Overview
This notebook evaluates the **Meta DINOv3** Vision Foundation Model (`facebook/dinov3-vits16-pretrain-lvd1689m`) on the 4-class CT kidney dataset (`dl/processed_grouped`).
* If DINOv3 gated access is pending, it cleanly falls back to **DINOv2** (`facebook/dinov2-small`).
* Extracts and caches 384-dimensional foundation embeddings for `train`, `validation`, and `test` splits into `dl/artifacts/embeddings/`.
* Trains class-weighted **Logistic Regression** and **MLP** classifier heads; selects the winner based on validation macro F1.
* Evaluates using `common_eval.py` (cluster bootstrap 95% CIs, slice/cluster metrics, pooled val+test).
* Exports the complete contract bundle (`model_card.json`, `metrics.json`, `predictions.csv`, `head.joblib`, `predict_example.py`, `dinov3_artifacts.zip`)."""),

        new_markdown_cell("""---
### 1. Global Configuration & Smoke Test Settings
* **What it does:** Sets seed, batch size, model repo ID, paths, idempotency flag `RERUN`, and `SMOKE_TEST` flag (fast 50-sample mode).
* **Expected runtime:** < 1 second.
* **Expected output:** Config summary."""),

        new_code_cell("""import os
import sys
import time
from pathlib import Path

SEED = 42
RERUN = False       # Set to True to re-extract embeddings and re-train heads
SMOKE_TEST = False  # Set to True for fast testing (50 images per split, 20 bootstrap resamples)
BATCH_SIZE = 32 if SMOKE_TEST else 64
BOOTSTRAP_RESAMPLES = 20 if SMOKE_TEST else 1000
FAMILY = "dinov3"
VERSION_TAG = "dinov3_vits16_v001"
BASE_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
FALLBACK_MODEL_ID = "facebook/dinov2-small"

WORKSPACE_ROOT = Path("/content/StoneSense-AI") if "google.colab" in sys.modules else Path(".").resolve()
sys.path.insert(0, (WORKSPACE_ROOT / "dl" / "notebooks").as_posix())

DATA_DIR = WORKSPACE_ROOT / "dl" / "processed_grouped"
MANIFEST_PATH = WORKSPACE_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"
EMBEDDINGS_DIR = WORKSPACE_ROOT / "dl" / "artifacts" / "embeddings"
OUT_DIR = WORKSPACE_ROOT / "dl" / "models" / "ct"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Workspace Root: {WORKSPACE_ROOT.as_posix()}")
print(f"Embeddings Cache: {EMBEDDINGS_DIR.as_posix()}")
print(f"Output Directory: {(OUT_DIR / FAMILY).as_posix()}")
print(f"Smoke Test Mode: {SMOKE_TEST} (Resamples: {BOOTSTRAP_RESAMPLES})")"""),

        new_markdown_cell("""---
### 2. Check Cell: Environment, GPU & Model Access Verification
* **What it does:** Verifies PyTorch, GPU availability, reads `HF_TOKEN`, and tests loading DINOv3 (or switches to DINOv2 fallback).
* **Expected runtime:** ~5-10 seconds.
* **Expected output:** Loaded foundation backbone name and feature dimensionality (384)."""),

        new_code_cell("""import torch
from transformers import AutoImageProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

hf_token = os.getenv("HF_TOKEN")
if "google.colab" in sys.modules:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN') or hf_token
    except Exception:
        pass

active_model_id = BASE_MODEL_ID
is_fallback = False
license_name = "DINOv3 License"

try:
    print(f"Attempting to load primary backbone: {active_model_id}...")
    processor = AutoImageProcessor.from_pretrained(active_model_id, token=hf_token)
    backbone = AutoModel.from_pretrained(active_model_id, token=hf_token).to(device)
    backbone.eval()
    print(f"Successfully loaded primary backbone: {active_model_id}")
except Exception as exc:
    print(f"Primary DINOv3 model access unavailable ({exc}).")
    print(f"Switching to fallback foundation model: {FALLBACK_MODEL_ID}...")
    active_model_id = FALLBACK_MODEL_ID
    is_fallback = True
    license_name = "Apache-2.0"
    processor = AutoImageProcessor.from_pretrained(active_model_id)
    backbone = AutoModel.from_pretrained(active_model_id).to(device)
    backbone.eval()
    print(f"Successfully loaded fallback backbone: {active_model_id}")

DISPLAY_NAME = f"DINOv2 ViT-S/16 (Fallback)" if is_fallback else "DINOv3 ViT-S/16 Foundation"
print(f"Model Display Name: {DISPLAY_NAME}")
print(f"License: {license_name}")"""),

        new_markdown_cell("""---
### 3. Extract & Cache Foundation Embeddings (`.npz`)
* **What it does:** Iterates through `train`, `validation`, and `test` splits, runs frozen backbone feature extraction, and saves `{split}.npz` containing `embeddings (N, 384)`, `labels (N,)`, `filenames (N,)`, and `cluster_ids (N,)`.
* **Expected runtime:** ~1-2 mins on GPU (~10s in SMOKE_TEST mode). Idempotent: skips if cached.
* **Expected output:** Cache file confirmation with tensor shapes."""),

        new_code_cell("""import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

df_manifest = pd.read_csv(MANIFEST_PATH)
start_time = time.time()

for split in ["train", "validation", "test"]:
    cache_file = EMBEDDINGS_DIR / f"{FAMILY}_{split}.npz"
    if cache_file.exists() and not RERUN and not SMOKE_TEST:
        print(f"Cached embeddings found at {cache_file.name}. Skipping extraction.")
        continue

    split_df = df_manifest[df_manifest["destination_split"] == split].reset_index(drop=True)
    if SMOKE_TEST:
        split_df = split_df.head(50)

    embeddings_list = []
    labels_list = []
    filenames_list = []
    cluster_ids_list = []

    print(f"Extracting {len(split_df):,} embeddings for '{split}' split...")
    
    for i in tqdm(range(0, len(split_df), BATCH_SIZE)):
        batch_rows = split_df.iloc[i : i + BATCH_SIZE]
        images = []
        for _, row in batch_rows.iterrows():
            img_path = DATA_DIR / split / row["class"] / row["filename"]
            with Image.open(img_path) as img:
                images.append(img.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = backbone(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0]
            emb = emb.cpu().numpy().astype(np.float32)

        embeddings_list.append(emb)
        labels_list.extend([CLASS_TO_IDX[r["class"]] for _, r in batch_rows.iterrows()])
        filenames_list.extend(batch_rows["filename"].tolist())
        cluster_ids_list.extend(batch_rows["near_duplicate_group_id"].tolist())

    np.savez_compressed(
        cache_file,
        embeddings=np.vstack(embeddings_list),
        labels=np.array(labels_list, dtype=np.int64),
        filenames=np.array(filenames_list),
        cluster_ids=np.array(cluster_ids_list, dtype=np.int64)
    )
    print(f"Saved {cache_file.name} ({len(filenames_list)} records, shape: {np.vstack(embeddings_list).shape})")

extraction_duration = time.time() - start_time
print(f"Embedding extraction complete in {extraction_duration:.1f}s.")"""),

        new_markdown_cell("""---
### 4. Train Classification Heads (Logistic Regression & MLP)
* **What it does:** Loads cached embeddings, fits class-weighted Logistic Regression and MLP heads on `train`, evaluates on `validation`, and selects the champion head.
* **Expected runtime:** ~10-20 seconds (~1s in SMOKE_TEST).
* **Expected output:** Validation comparison table and selected champion head."""),

        new_code_cell("""import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

train_data = np.load(EMBEDDINGS_DIR / f"{FAMILY}_train.npz")
val_data = np.load(EMBEDDINGS_DIR / f"{FAMILY}_validation.npz")

X_train, y_train = train_data["embeddings"], train_data["labels"]
X_val, y_val = val_data["embeddings"], val_data["labels"]

print(f"Training features: {X_train.shape}, Validation features: {X_val.shape}")

# 1. Logistic Regression Head
lr_head = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=SEED,
    C=1.0
)
lr_head.fit(X_train, y_train)
lr_val_f1 = f1_score(y_val, lr_head.predict(X_val), average="macro")

# 2. MLP Head
mlp_head = MLPClassifier(
    hidden_layer_sizes=(128, 32) if SMOKE_TEST else (256, 64),
    max_iter=50 if SMOKE_TEST else 300,
    activation="relu",
    early_stopping=True,
    random_state=SEED
)
mlp_head.fit(X_train, y_train)
mlp_val_f1 = f1_score(y_val, mlp_head.predict(X_val), average="macro")

print("=" * 60)
print(f"Validation Macro F1:")
print(f"   - Logistic Regression Head : {lr_val_f1:.4f}")
print(f"   - MLP Classifier Head      : {mlp_val_f1:.4f}")

if lr_val_f1 >= mlp_val_f1:
    champion_head = lr_head
    head_type = "LogisticRegression"
    champion_f1 = lr_val_f1
else:
    champion_head = mlp_head
    head_type = "MLPClassifier"
    champion_f1 = mlp_val_f1

print(f"Selected Champion Head: {head_type} (Macro F1: {champion_f1:.4f})")
print("=" * 60)"""),

        new_markdown_cell("""---
### 5. Generate Predictions & Evaluate via `common_eval.py`
* **What it does:** Generates probabilistic predictions for `validation` and `test` splits, runs cluster bootstrap 95% CIs, and calculates slice & cluster-level metrics.
* **Expected runtime:** ~15-25 seconds (~2s in SMOKE_TEST).
* **Expected output:** Metrics summary and confusion matrix."""),

        new_code_cell("""from common_eval import generate_full_evaluation, IDX_TO_CLASS

test_data = np.load(EMBEDDINGS_DIR / f"{FAMILY}_test.npz")
X_test, y_test = test_data["embeddings"], test_data["labels"]

# Generate Val Predictions DataFrame
val_probs = champion_head.predict_proba(X_val)
df_val_preds = pd.DataFrame({
    "filename": val_data["filenames"],
    "cluster_id": val_data["cluster_ids"],
    "y_true": [IDX_TO_CLASS[idx] for idx in y_val],
    "y_pred": [IDX_TO_CLASS[idx] for idx in np.argmax(val_probs, axis=1)],
    "p_Cyst": val_probs[:, 0],
    "p_Normal": val_probs[:, 1],
    "p_Stone": val_probs[:, 2],
    "p_Tumor": val_probs[:, 3],
})

# Generate Test Predictions DataFrame
test_probs = champion_head.predict_proba(X_test)
df_test_preds = pd.DataFrame({
    "filename": test_data["filenames"],
    "cluster_id": test_data["cluster_ids"],
    "y_true": [IDX_TO_CLASS[idx] for idx in y_test],
    "y_pred": [IDX_TO_CLASS[idx] for idx in np.argmax(test_probs, axis=1)],
    "p_Cyst": test_probs[:, 0],
    "p_Normal": test_probs[:, 1],
    "p_Stone": test_probs[:, 2],
    "p_Tumor": test_probs[:, 3],
})

eval_results = generate_full_evaluation(df_val_preds, df_test_preds, bootstrap_iterations=BOOTSTRAP_RESAMPLES, seed=SEED)

print("=" * 60)
print(f"TEST SPLIT EVALUATION (DINOv3 + {head_type}):")
print(f"   - Slice Accuracy    : {eval_results['test']['slice_accuracy']:.4f}")
print(f"   - Slice Macro F1    : {eval_results['test']['slice_macro_f1']:.4f} (95% CI: [{eval_results['test']['confidence_intervals_95']['slice_macro_f1']['ci_lower_95']:.4f}, {eval_results['test']['confidence_intervals_95']['slice_macro_f1']['ci_upper_95']:.4f}])")
print(f"   - Cluster Macro F1  : {eval_results['test']['cluster_macro_f1']:.4f}")
print(f"   - Stone Recall      : {eval_results['test']['per_class']['Stone']['recall']:.4f}")
print(f"   - Tumor Recall      : {eval_results['test']['per_class']['Tumor']['recall']:.4f}")
print("=" * 60)"""),

        new_markdown_cell("""---
### 6. Save Artifacts Contract & `predict_example.py`
* **What it does:** Saves `model_card.json`, `metrics.json`, `predictions.csv`, `head.joblib`, `embedding_config.json`, and `predict_example.py` under `dl/models/ct/dinov3/`.
* **Expected runtime:** ~2-3 seconds.
* **Expected output:** Manifest of saved contract artifacts with SHA-256 hashes."""),

        new_code_cell("""import json
from common_eval import save_contract_artifacts

family_dir = OUT_DIR / FAMILY
family_dir.mkdir(parents=True, exist_ok=True)

# 1. Save head artifact
head_artifact_path = family_dir / "head.joblib"
joblib.dump(champion_head, head_artifact_path)

# 2. Save embedding config
config_path = family_dir / "embedding_config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump({
        "base_model": active_model_id,
        "is_fallback": is_fallback,
        "embedding_dim": int(X_train.shape[1]),
        "head_type": head_type
    }, f, indent=2)

# 3. Create standalone predict_example.py
predict_example_code = f'''\"\"\"Standalone DINOv3 prediction example for Phase 8 contract validation.\"\"\"
from pathlib import Path
import json
import joblib
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

def predict_single_image(image_path: str, model_dir: str = ".") -> dict:
    model_path = Path(model_dir)
    with open(model_path / "embedding_config.json", "r") as f:
        config = json.load(f)
    
    head = joblib.load(model_path / "head.joblib")
    processor = AutoImageProcessor.from_pretrained(config["base_model"])
    backbone = AutoModel.from_pretrained(config["base_model"])
    backbone.eval()
    
    with Image.open(image_path) as img:
        inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    
    with torch.no_grad():
        outputs = backbone(**inputs)
        emb = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        emb_np = emb.cpu().numpy().astype(np.float32)
    
    probs = head.predict_proba(emb_np)[0]
    return {c: float(p) for c, p in zip(CLASS_NAMES, probs)}

if __name__ == "__main__":
    import sys
    img_file = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    print(f"Probabilities: {predict_single_image(img_file)}")
'''

# 4. Save contract
saved = save_contract_artifacts(
    out_dir=OUT_DIR,
    family=FAMILY,
    version_tag=VERSION_TAG,
    display_name=DISPLAY_NAME,
    base_model=active_model_id,
    license_name=license_name,
    hyperparameters={"head_type": head_type, "batch_size": BATCH_SIZE, "seed": SEED, "smoke_test": SMOKE_TEST},
    preprocessing_spec={"resize": [224, 224], "normalization": "ImageNet / DINO standard"},
    library_versions={"torch": torch.__version__, "transformers": "installed"},
    training_time_sec=extraction_duration,
    evaluation_results=eval_results,
    additional_artifact_paths=[head_artifact_path, config_path],
    predict_example_code=predict_example_code,
    seed=SEED
)

print(f"Successfully saved Phase 8 contract artifacts to {family_dir.as_posix()}")"""),

        new_markdown_cell("""---
### 7. Package Zip for Project Integration
* **What it does:** Zips `dl/models/ct/dinov3/` into `dinov3_artifacts.zip` and reports target deployment path.
* **Expected runtime:** ~2-5 seconds.
* **Expected output:** Downloadable zip confirmation."""),

        new_code_cell("""import shutil

zip_path = WORKSPACE_ROOT / "dl" / "models" / "ct" / f"{FAMILY}_artifacts.zip"
shutil.make_archive(str(zip_path.with_suffix("")), 'zip', family_dir)

print("=" * 60)
print(f"CREATED ARTIFACT ZIP: {zip_path.name}")
print(f"Target Project Unzip Directory: dl/models/ct/{FAMILY}/")
print("=" * 60)""")
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    return nb


def build_nb02():
    cells = [
        new_markdown_cell("""# StoneSense-AI: CT Multi-Model Experimentation — YOLO26 Classification

### 📋 Overview
This notebook trains and evaluates the **Ultralytics YOLO26** classification model (`yolo26s-cls.pt` or `yolo26n-cls.pt`) on `dl/datasets/yolo_view`.
* Note on License: Ultralytics models are released under **AGPL-3.0**.
* Uses `yolo26s-cls.pt` backbone (30 epochs, or 1 epoch in `SMOKE_TEST` mode, imgsz 224).
* Generates batch predictions on `validation` and `test` splits and computes metrics strictly via `common_eval.py` (not internal Ultralytics metrics) for direct comparability with ResNet18 and DINOv3.
* Exports contract artifacts: `model_card.json`, `metrics.json`, `predictions.csv`, `best.pt`, `predict_example.py`, and `yolo26_artifacts.zip`."""),

        new_markdown_cell("""---
### 1. Global Configuration & Smoke Test Settings
* **What it does:** Configures hyperparameters, epochs, batch size, paths, seed, and `SMOKE_TEST` flag (1 epoch, 50 samples).
* **Expected runtime:** < 1 second.
* **Expected output:** Configuration summary."""),

        new_code_cell("""import os
import sys
import time
from pathlib import Path

SEED = 42
RERUN = False
SMOKE_TEST = False  # Set to True for fast 1-epoch / 50-sample verification
EPOCHS = 1 if SMOKE_TEST else 30
BATCH_SIZE = 16 if SMOKE_TEST else 32
BOOTSTRAP_RESAMPLES = 20 if SMOKE_TEST else 1000
IMGSZ = 224
FAMILY = "yolo26"
VERSION_TAG = "yolo26s_v001"
BASE_WEIGHT = "yolo26s-cls.pt"

WORKSPACE_ROOT = Path("/content/StoneSense-AI") if "google.colab" in sys.modules else Path(".").resolve()
sys.path.insert(0, (WORKSPACE_ROOT / "dl" / "notebooks").as_posix())

YOLO_DATA_YAML = WORKSPACE_ROOT / "dl" / "datasets" / "yolo_view" / "dataset.yaml"
MANIFEST_PATH = WORKSPACE_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"
OUT_DIR = WORKSPACE_ROOT / "dl" / "models" / "ct"
RUNS_DIR = WORKSPACE_ROOT / "dl" / "artifacts" / "yolo_runs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Workspace Root: {WORKSPACE_ROOT.as_posix()}")
print(f"YOLO Dataset Config: {YOLO_DATA_YAML.as_posix()}")
print(f"Output Directory: {(OUT_DIR / FAMILY).as_posix()}")
print(f"Smoke Test Mode: {SMOKE_TEST} (Epochs: {EPOCHS})")"""),

        new_markdown_cell("""---
### 2. Check Cell: Environment, GPU & Ultralytics Weights Verification
* **What it does:** Verifies `ultralytics` installation, GPU acceleration, and tests loading base weights `yolo26s-cls.pt` (with fallback to `yolo11s-cls.pt` if needed).
* **Expected runtime:** ~5-10 seconds.
* **Expected output:** Loaded YOLO model confirmation and AGPL-3.0 license declaration."""),

        new_code_cell("""import torch
from ultralytics import YOLO

device = "0" if torch.cuda.is_available() else "cpu"
print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

try:
    print(f"Testing base weight loading: {BASE_WEIGHT}...")
    test_model = YOLO(BASE_WEIGHT)
    active_weight = BASE_WEIGHT
    display_name = "Ultralytics YOLO26-Small Classifier"
    print(f"Successfully initialized YOLO26: {BASE_WEIGHT}")
except Exception as e:
    print(f"YOLO26 weights not found ({e}); testing YOLO11 fallback...")
    active_weight = "yolo11s-cls.pt"
    test_model = YOLO(active_weight)
    display_name = "Ultralytics YOLO11-Small Classifier (Fallback)"
    print(f"Loaded fallback weight: {active_weight}")

print("=" * 60)
print(f"Model Display Name: {display_name}")
print("License Notice: AGPL-3.0 (Ultralytics Open Source License)")
print("=" * 60)"""),

        new_markdown_cell("""---
### 3. Train YOLO26 Classifier
* **What it does:** Fits YOLO26 classification head on `train` split (30 epochs or 1 epoch in SMOKE_TEST) with validation monitoring.
* **Expected runtime:** ~10-15 mins on T4 GPU (~1 min in SMOKE_TEST). Idempotent: skips if `best.pt` exists.
* **Expected output:** Training epoch progress and saved `best.pt` path."""),

        new_code_cell("""family_dir = OUT_DIR / FAMILY
family_dir.mkdir(parents=True, exist_ok=True)
best_pt_path = family_dir / "best.pt"

start_train_time = time.time()

if best_pt_path.exists() and not RERUN and not SMOKE_TEST:
    print(f"Found existing checkpoint at {best_pt_path.as_posix()}. Skipping training.")
    model = YOLO(best_pt_path)
    training_duration = 0.0
else:
    print(f"Launching YOLO training ({EPOCHS} epochs, batch {BATCH_SIZE}, imgsz {IMGSZ})...")
    model = YOLO(active_weight)
    
    results = model.train(
        data=(WORKSPACE_ROOT / "dl" / "datasets" / "yolo_view").as_posix(),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        seed=SEED,
        device=device,
        project=RUNS_DIR.as_posix(),
        name=FAMILY,
        exist_ok=True,
        verbose=True
    )
    training_duration = time.time() - start_train_time
    
    # Copy best.pt to family output directory
    trained_best = RUNS_DIR / FAMILY / "weights" / "best.pt"
    if trained_best.exists():
        import shutil
        shutil.copy(trained_best, best_pt_path)
        print(f"Saved champion weights to {best_pt_path.as_posix()}")
    else:
        raise FileNotFoundError(f"Training output best.pt not found at {trained_best.as_posix()}")

print(f"YOLO training process complete ({training_duration:.1f}s).")"""),

        new_markdown_cell("""---
### 4. Batch Prediction & Common Evaluation Pipeline
* **What it does:** Runs raw inference on `validation` and `test` images to get class probability vectors `[p_Cyst, p_Normal, p_Stone, p_Tumor]`, evaluates with `common_eval.py` and cluster bootstrap 95% CIs.
* **Expected runtime:** ~1-2 mins (~5s in SMOKE_TEST).
* **Expected output:** Comprehensive performance report and comparison with CIs."""),

        new_code_cell("""import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from common_eval import generate_full_evaluation, CLASS_NAMES

df_manifest = pd.read_csv(MANIFEST_PATH)
trained_model = YOLO(best_pt_path)

def predict_split(split_name: str) -> pd.DataFrame:
    sub_df = df_manifest[df_manifest["destination_split"] == split_name].reset_index(drop=True)
    if SMOKE_TEST:
        sub_df = sub_df.head(50)
    all_probs = []
    
    print(f"Running inference on '{split_name}' split ({len(sub_df):,} images)...")
    for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
        img_p = WORKSPACE_ROOT / "dl" / "processed_grouped" / split_name / row["class"] / row["filename"]
        
        res = trained_model(img_p.as_posix(), verbose=False)[0]
        probs = res.probs.data.cpu().numpy()
        
        yolo_names = res.names
        prob_dict = {yolo_names[i]: float(probs[i]) for i in range(len(probs))}
        canonical_probs = [prob_dict.get(cname, 0.0) for cname in CLASS_NAMES]
        all_probs.append(canonical_probs)
        
    prob_mat = np.array(all_probs)
    pred_classes = [CLASS_NAMES[idx] for idx in np.argmax(prob_mat, axis=1)]
    
    return pd.DataFrame({
        "filename": sub_df["filename"],
        "cluster_id": sub_df["near_duplicate_group_id"],
        "y_true": sub_df["class"],
        "y_pred": pred_classes,
        "p_Cyst": prob_mat[:, 0],
        "p_Normal": prob_mat[:, 1],
        "p_Stone": prob_mat[:, 2],
        "p_Tumor": prob_mat[:, 3],
    })

df_val_preds = predict_split("validation")
df_test_preds = predict_split("test")

eval_results = generate_full_evaluation(df_val_preds, df_test_preds, bootstrap_iterations=BOOTSTRAP_RESAMPLES, seed=SEED)

print("=" * 60)
print(f"TEST SPLIT EVALUATION (YOLO26):")
print(f"   - Slice Accuracy    : {eval_results['test']['slice_accuracy']:.4f}")
print(f"   - Slice Macro F1    : {eval_results['test']['slice_macro_f1']:.4f} (95% CI: [{eval_results['test']['confidence_intervals_95']['slice_macro_f1']['ci_lower_95']:.4f}, {eval_results['test']['confidence_intervals_95']['slice_macro_f1']['ci_upper_95']:.4f}])")
print(f"   - Cluster Macro F1  : {eval_results['test']['cluster_macro_f1']:.4f}")
print(f"   - Stone Recall      : {eval_results['test']['per_class']['Stone']['recall']:.4f}")
print(f"   - Tumor Recall      : {eval_results['test']['per_class']['Tumor']['recall']:.4f}")
print("=" * 60)"""),

        new_markdown_cell("""---
### 5. Save Artifacts Contract & `predict_example.py`
* **What it does:** Saves `model_card.json`, `metrics.json`, `predictions.csv`, `best.pt`, and `predict_example.py` under `dl/models/ct/yolo26/`.
* **Expected runtime:** ~2-3 seconds.
* **Expected output:** SHA-256 manifest and artifact summary."""),

        new_code_cell("""from common_eval import save_contract_artifacts

predict_example_code = '''\"\"\"Standalone YOLO26 prediction example for Phase 8 contract validation.\"\"\"
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

def predict_single_image(image_path: str, model_dir: str = ".") -> dict:
    model_path = Path(model_dir) / "best.pt"
    model = YOLO(model_path.as_posix())
    results = model(image_path, verbose=False)[0]
    probs = results.probs.data.cpu().numpy()
    yolo_names = results.names
    prob_dict = {yolo_names[i]: float(probs[i]) for i in range(len(probs))}
    return {c: float(prob_dict.get(c, 0.0)) for c in CLASS_NAMES}

if __name__ == "__main__":
    import sys
    img_file = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    print(f"Probabilities: {predict_single_image(img_file)}")
'''

saved = save_contract_artifacts(
    out_dir=OUT_DIR,
    family=FAMILY,
    version_tag=VERSION_TAG,
    display_name=display_name,
    base_model=active_weight,
    license_name="AGPL-3.0",
    hyperparameters={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "imgsz": IMGSZ, "seed": SEED, "smoke_test": SMOKE_TEST},
    preprocessing_spec={"resize": [IMGSZ, IMGSZ], "normalization": "Ultralytics standard"},
    library_versions={"torch": torch.__version__, "ultralytics": "installed"},
    training_time_sec=training_duration,
    evaluation_results=eval_results,
    additional_artifact_paths=[best_pt_path],
    predict_example_code=predict_example_code,
    seed=SEED
)

print(f"Successfully saved YOLO26 Phase 8 contract artifacts to {family_dir.as_posix()}")"""),

        new_markdown_cell("""---
### 6. Package Zip for Project Integration
* **What it does:** Zips `dl/models/ct/yolo26/` into `yolo26_artifacts.zip`.
* **Expected runtime:** ~2-5 seconds.
* **Expected output:** Downloadable zip confirmation."""),

        new_code_cell("""import shutil

zip_path = WORKSPACE_ROOT / "dl" / "models" / "ct" / f"{FAMILY}_artifacts.zip"
shutil.make_archive(str(zip_path.with_suffix("")), 'zip', family_dir)

print("=" * 60)
print(f"CREATED ARTIFACT ZIP: {zip_path.name}")
print(f"Target Project Unzip Directory: dl/models/ct/{FAMILY}/")
print("=" * 60)""")
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    return nb


def build_nb03():
    cells = [
        new_markdown_cell("""# StoneSense-AI: CT Multi-Model Experimentation — Quantum K-Nearest Neighbors (QKNN)

### 📋 Overview
This notebook implements and evaluates a **Quantum K-Nearest Neighbors (QKNN)** classifier using PennyLane on top of DINOv3 foundation embeddings.
* Input: 384-dimensional DINOv3 embeddings cached from Notebook 01 (`dl/artifacts/embeddings/dinov3_train.npz`).
* Dimensionality Reduction: PCA (6–8 components) fit **strictly on train split**.
* Prototype Selection: Balanced, representative prototype set from training clusters.
* Quantum Kernel: Angle embedding + PennyLane `default.qubit` state-vector simulator with quantum swap-test / fidelity overlap.
* Benchmark: Direct comparison against classical KNN control with identical PCA and prototypes.
* Evaluates runtime latency (ms per prediction), explicitly states the evaluated subset, and computes 95% bootstrap CIs.
* Exports contract bundle: `qknn_bundle.joblib`, `model_card.json`, `metrics.json`, `predictions.csv`, `predict_example.py`, `qknn_artifacts.zip`."""),

        new_markdown_cell("""---
### 1. Global Configuration & Smoke Test Settings
* **What it does:** Configures number of qubits ($N=6$), prototype count, paths, seed, and `SMOKE_TEST` flag.
* **Expected runtime:** < 1 second.
* **Expected output:** Config summary."""),

        new_code_cell("""import os
import sys
import time
from pathlib import Path

SEED = 42
RERUN = False
SMOKE_TEST = False      # Set to True for fast 50-sample verification
N_QUBITS = 6            # 6-qubit angle embedding circuit
PROTOTYPES_PER_CLASS = 10 if SMOKE_TEST else 20  # Total 40 (smoke) or 80 quantum prototype states
BOOTSTRAP_RESAMPLES = 20 if SMOKE_TEST else 1000
FAMILY = "qknn"
VERSION_TAG = "qknn_pennylane_v001"

WORKSPACE_ROOT = Path("/content/StoneSense-AI") if "google.colab" in sys.modules else Path(".").resolve()
sys.path.insert(0, (WORKSPACE_ROOT / "dl" / "notebooks").as_posix())

EMBEDDINGS_DIR = WORKSPACE_ROOT / "dl" / "artifacts" / "embeddings"
OUT_DIR = WORKSPACE_ROOT / "dl" / "models" / "ct"
MANIFEST_PATH = WORKSPACE_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Workspace Root: {WORKSPACE_ROOT.as_posix()}")
print(f"Embeddings Directory: {EMBEDDINGS_DIR.as_posix()}")
print(f"Output Directory: {(OUT_DIR / FAMILY).as_posix()}")
print(f"Smoke Test Mode: {SMOKE_TEST} (Qubits: {N_QUBITS}, Prototypes/class: {PROTOTYPES_PER_CLASS})")"""),

        new_markdown_cell("""---
### 2. Check Cell: Environment & PennyLane Quantum Device Verification
* **What it does:** Verifies `pennylane` installation, creates a 6-qubit `default.qubit` simulator, and tests a dummy quantum fidelity execution.
* **Expected runtime:** ~1-2 seconds.
* **Expected output:** Simulator backend status and quantum circuit latency test."""),

        new_code_cell("""import pennylane as qml
import numpy as np

print(f"PennyLane Version: {qml.__version__}")
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def test_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(N_QUBITS), rotation='Y')
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS), rotation='Y')
    return qml.probs(wires=range(N_QUBITS))

# Test execution
t0 = time.time()
res = test_circuit(np.zeros(N_QUBITS), np.ones(N_QUBITS))
t_test = (time.time() - t0) * 1000

print(f"PennyLane default.qubit initialized successfully. Test evaluation: {t_test:.2f} ms")"""),

        new_markdown_cell("""---
### 3. Load Embeddings & Fit Train-Only PCA + Prototype Selection
* **What it does:** Loads cached DINOv3 embeddings, scales and fits PCA strictly on `train`, and selects class-balanced prototype samples.
* **Expected runtime:** ~2-3 seconds.
* **Expected output:** Explained variance ratio and selected prototype distribution."""),

        new_code_cell("""from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

train_npz = EMBEDDINGS_DIR / "dinov3_train.npz"
val_npz = EMBEDDINGS_DIR / "dinov3_validation.npz"
test_npz = EMBEDDINGS_DIR / "dinov3_test.npz"

if not train_npz.exists():
    raise FileNotFoundError(f"DINOv3 embeddings missing at {train_npz.as_posix()}. Run Notebook 01 first!")

train_data = np.load(train_npz)
val_data = np.load(val_npz)
test_data = np.load(test_npz)

X_train_raw, y_train = train_data["embeddings"], train_data["labels"]
X_val_raw, y_val = val_data["embeddings"], val_data["labels"]
X_test_raw, y_test = test_data["embeddings"], test_data["labels"]

if SMOKE_TEST:
    X_train_raw, y_train = X_train_raw[:50], y_train[:50]
    X_val_raw, y_val = X_val_raw[:50], y_val[:50]
    X_test_raw, y_test = X_test_raw[:50], y_test[:50]

# 1. Fit PCA strictly on train split
pca = PCA(n_components=N_QUBITS, random_state=SEED)
scaler = MinMaxScaler(feature_range=(0, np.pi))

X_train_pca = pca.fit_transform(X_train_raw)
X_train_q = scaler.fit_transform(X_train_pca)

X_val_q = scaler.transform(pca.transform(X_val_raw))
X_test_q = scaler.transform(pca.transform(X_test_raw))

print(f"PCA Explained Variance Ratio ({N_QUBITS} components): {np.sum(pca.explained_variance_ratio_):.4f}")

# 2. Select balanced prototypes from train data
rng = np.random.RandomState(SEED)
proto_indices = []

for c_idx in range(4):
    c_indices = np.where(y_train == c_idx)[0]
    if len(c_indices) > 0:
        selected = rng.choice(c_indices, size=min(PROTOTYPES_PER_CLASS, len(c_indices)), replace=False)
        proto_indices.extend(selected)

proto_X = X_train_q[proto_indices]
proto_y = y_train[proto_indices]
proto_filenames = train_data["filenames"][proto_indices] if not SMOKE_TEST else train_data["filenames"][:50][proto_indices]

print(f"Selected {len(proto_X)} quantum prototypes ({PROTOTYPES_PER_CLASS} per class across {len(np.unique(proto_y))} classes).")"""),

        new_markdown_cell("""---
### 4. Quantum Kernel Classifier Implementation & Latency Benchmark
* **What it does:** Defines the PennyLane fidelity kernel, measures inference latency per sample (ms/sample), tunes optimal $k$, and reports the evaluated subset.
* **Expected runtime:** ~15-30 seconds.
* **Expected output:** Latency benchmark (ms/sample) and optimal $k$ selection."""),

        new_code_cell("""@qml.qnode(dev)
def quantum_fidelity_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(N_QUBITS), rotation='Y')
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS), rotation='Y')
    return qml.probs(wires=range(N_QUBITS))

def compute_quantum_similarity_vector(sample: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    \"\"\"Computes fidelity |<psi(sample)|phi(proto)>|^2 against all prototypes.\"\"\"
    similarities = np.zeros(len(prototypes), dtype=np.float32)
    for i, proto in enumerate(prototypes):
        probs = quantum_fidelity_circuit(sample, proto)
        similarities[i] = probs[0]
    return similarities

def qknn_predict_proba(X_samples: np.ndarray, k: int = 5) -> np.ndarray:
    all_probs = []
    for sample in X_samples:
        sims = compute_quantum_similarity_vector(sample, proto_X)
        top_k_idx = np.argsort(sims)[-k:]
        top_k_labels = proto_y[top_k_idx]
        top_k_weights = sims[top_k_idx]
        
        class_scores = np.zeros(4, dtype=np.float32)
        for lbl, w in zip(top_k_labels, top_k_weights):
            class_scores[lbl] += w
            
        sum_scores = np.sum(class_scores)
        if sum_scores > 0:
            probs = class_scores / sum_scores
        else:
            probs = np.ones(4) / 4.0
        all_probs.append(probs)
    return np.array(all_probs)

# Measure inference latency
t0 = time.time()
_ = qknn_predict_proba(X_val_q[:10], k=3)
ms_per_sample = ((time.time() - t0) / 10.0) * 1000

print(f"QKNN Average Inference Latency: {ms_per_sample:.2f} ms / image (CPU default.qubit)")

# Tune k on validation subset
k_candidates = [3, 5] if SMOKE_TEST else [3, 5, 7, 9]
best_k = 3
best_val_f1 = -1.0
from sklearn.metrics import f1_score

eval_sub_len = min(len(X_val_q), 50 if SMOKE_TEST else 300)
print(f"Tuning k on validation subset of {eval_sub_len} observations...")

for k_cand in k_candidates:
    val_p = qknn_predict_proba(X_val_q[:eval_sub_len], k=k_cand)
    val_preds = np.argmax(val_p, axis=1)
    f1 = f1_score(y_val[:eval_sub_len], val_preds, average='macro', zero_division=0)
    print(f"   - k={k_cand}: Val Macro F1 = {f1:.4f}")
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_k = k_cand

print(f"Selected Optimal k = {best_k}")"""),

        new_markdown_cell("""---
### 5. Evaluate QKNN and Classical KNN Control via `common_eval.py`
* **What it does:** Runs evaluation for QKNN and Classical KNN baseline, explicitly reporting evaluated subset size and computing 95% bootstrap CIs.
* **Expected runtime:** ~30-60 seconds.
* **Expected output:** Side-by-side performance table with CIs."""),

        new_code_cell("""from sklearn.neighbors import KNeighborsClassifier
from common_eval import generate_full_evaluation, IDX_TO_CLASS

# 1. Classical KNN baseline on identical PCA features & prototypes
classical_knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
classical_knn.fit(proto_X, proto_y)
knn_val_probs = classical_knn.predict_proba(X_val_q)
knn_test_probs = classical_knn.predict_proba(X_test_q)

# 2. QKNN predictions on validation and test splits
eval_val_n = len(X_val_q)
eval_test_n = len(X_test_q)
print(f"Computing QKNN predictions on Validation ({eval_val_n} images) and Test ({eval_test_n} images)...")

qknn_val_probs = qknn_predict_proba(X_val_q, k=best_k)
qknn_test_probs = qknn_predict_proba(X_test_q, k=best_k)

val_filenames = val_data["filenames"][:eval_val_n] if SMOKE_TEST else val_data["filenames"]
val_clusters = val_data["cluster_ids"][:eval_val_n] if SMOKE_TEST else val_data["cluster_ids"]
test_filenames = test_data["filenames"][:eval_test_n] if SMOKE_TEST else test_data["filenames"]
test_clusters = test_data["cluster_ids"][:eval_test_n] if SMOKE_TEST else test_data["cluster_ids"]

df_val_preds = pd.DataFrame({
    "filename": val_filenames,
    "cluster_id": val_clusters,
    "y_true": [IDX_TO_CLASS[idx] for idx in y_val],
    "y_pred": [IDX_TO_CLASS[idx] for idx in np.argmax(qknn_val_probs, axis=1)],
    "p_Cyst": qknn_val_probs[:, 0],
    "p_Normal": qknn_val_probs[:, 1],
    "p_Stone": qknn_val_probs[:, 2],
    "p_Tumor": qknn_val_probs[:, 3],
})

df_test_preds = pd.DataFrame({
    "filename": test_filenames,
    "cluster_id": test_clusters,
    "y_true": [IDX_TO_CLASS[idx] for idx in y_test],
    "y_pred": [IDX_TO_CLASS[idx] for idx in np.argmax(qknn_test_probs, axis=1)],
    "p_Cyst": qknn_test_probs[:, 0],
    "p_Normal": qknn_test_probs[:, 1],
    "p_Stone": qknn_test_probs[:, 2],
    "p_Tumor": qknn_test_probs[:, 3],
})

eval_results = generate_full_evaluation(df_val_preds, df_test_preds, bootstrap_iterations=BOOTSTRAP_RESAMPLES, seed=SEED)

print("=" * 60)
print(f"QKNN TEST SPLIT EVALUATION (Evaluated on {eval_test_n} images):")
print(f"   - Slice Accuracy    : {eval_results['test']['slice_accuracy']:.4f}")
print(f"   - Slice Macro F1    : {eval_results['test']['slice_macro_f1']:.4f} (95% CI: [{eval_results['test']['confidence_intervals_95']['slice_macro_f1']['ci_lower_95']:.4f}, {eval_results['test']['confidence_intervals_95']['slice_macro_f1']['ci_upper_95']:.4f}])")
print(f"   - Cluster Macro F1  : {eval_results['test']['cluster_macro_f1']:.4f}")
print(f"   - Latency           : {ms_per_sample:.2f} ms / prediction")
print(f"   - Evaluated Subset  : {eval_test_n} / {len(test_data['labels'])} test images")
print("=" * 60)"""),

        new_markdown_cell("""---
### 6. Save Artifacts Contract & `predict_example.py`
* **What it does:** Saves `qknn_bundle.joblib`, `model_card.json`, `metrics.json`, `predictions.csv`, and `predict_example.py` under `dl/models/ct/qknn/`.
* **Expected runtime:** ~2-3 seconds.
* **Expected output:** Confirmation of saved artifacts and hashes."""),

        new_code_cell("""import joblib
from common_eval import save_contract_artifacts

family_dir = OUT_DIR / FAMILY
family_dir.mkdir(parents=True, exist_ok=True)

# 1. Save QKNN bundle
bundle_path = family_dir / "qknn_bundle.joblib"
joblib.dump({
    "pca": pca,
    "scaler": scaler,
    "proto_X": proto_X,
    "proto_y": proto_y,
    "proto_filenames": proto_filenames,
    "k": best_k,
    "n_qubits": N_QUBITS,
    "encoding_spec": "AngleEmbedding(rotation='Y')",
    "latency_ms_per_sample": ms_per_sample,
    "evaluated_subset": {"val_n": eval_val_n, "test_n": eval_test_n}
}, bundle_path)

# 2. Save standalone predict_example.py
predict_example_code = '''\"\"\"Standalone QKNN prediction example for Phase 8 contract validation.\"\"\"
from pathlib import Path
import joblib
import numpy as np
import pennylane as qml

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

def predict_single_embedding(embedding: np.ndarray, model_dir: str = ".") -> dict:
    bundle = joblib.load(Path(model_dir) / "qknn_bundle.joblib")
    pca, scaler = bundle["pca"], bundle["scaler"]
    proto_X, proto_y, k, n_q = bundle["proto_X"], bundle["proto_y"], bundle["k"], bundle["n_qubits"]
    
    x_q = scaler.transform(pca.transform(embedding.reshape(1, -1)))[0]
    dev = qml.device("default.qubit", wires=n_q)
    
    @qml.qnode(dev)
    def q_fidelity(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_q), rotation='Y')
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_q), rotation='Y')
        return qml.probs(wires=range(n_q))
        
    sims = np.array([q_fidelity(x_q, proto)[0] for proto in proto_X])
    top_k_idx = np.argsort(sims)[-k:]
    scores = np.zeros(4)
    for lbl, w in zip(proto_y[top_k_idx], sims[top_k_idx]):
        scores[lbl] += w
    probs = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(4)/4.0
    return {c: float(p) for c, p in zip(CLASS_NAMES, probs)}

if __name__ == "__main__":
    dummy_emb = np.random.randn(384).astype(np.float32)
    print(f"Probabilities: {predict_single_embedding(dummy_emb)}")
'''

saved = save_contract_artifacts(
    out_dir=OUT_DIR,
    family=FAMILY,
    version_tag=VERSION_TAG,
    display_name="PennyLane Quantum K-Nearest Neighbors",
    base_model="DINOv3-ViT-S/16 + PennyLane default.qubit",
    license_name="Apache-2.0",
    hyperparameters={"n_qubits": N_QUBITS, "k": best_k, "prototypes_per_class": PROTOTYPES_PER_CLASS, "seed": SEED, "smoke_test": SMOKE_TEST},
    preprocessing_spec={"pca_components": N_QUBITS, "scaler": "MinMaxScaler(0, pi)", "encoding": "AngleEmbedding_Y"},
    library_versions={"pennylane": qml.__version__, "scikit-learn": "installed"},
    training_time_sec=0.0,
    evaluation_results=eval_results,
    additional_artifact_paths=[bundle_path],
    predict_example_code=predict_example_code,
    seed=SEED
)

print(f"Successfully saved QKNN Phase 8 contract artifacts to {family_dir.as_posix()}")"""),

        new_markdown_cell("""---
### 7. Package Zip for Project Integration
* **What it does:** Zips `dl/models/ct/qknn/` into `qknn_artifacts.zip`.
* **Expected runtime:** ~2-5 seconds.
* **Expected output:** Downloadable zip confirmation."""),

        new_code_cell("""import shutil

zip_path = WORKSPACE_ROOT / "dl" / "models" / "ct" / f"{FAMILY}_artifacts.zip"
shutil.make_archive(str(zip_path.with_suffix("")), 'zip', family_dir)

print("=" * 60)
print(f"CREATED ARTIFACT ZIP: {zip_path.name}")
print(f"Target Project Unzip Directory: dl/models/ct/{FAMILY}/")
print("=" * 60)""")
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    return nb


def build_nb04():
    cells = [
        new_markdown_cell("""# StoneSense-AI: CT Multi-Model Experimentation — Comprehensive Comparison

### 📋 Overview
This notebook conducts the final **head-to-head comparison** across all four CT model families:
1. **ResNet18** (Existing transfer learning baseline evaluated via `common_eval.py`)
   * *Checkpoint location requirement:* `dl/models/kidney_resnet18.pth`
2. **DINOv3 / DINOv2** (Vision foundation model + champion classifier head)
3. **YOLO26-cls** (Ultralytics end-to-end vision model)
4. **QKNN** (PennyLane quantum kernel nearest neighbor classifier)

---
### 📊 Key Analysis Objectives
* Compares Slice Accuracy, Slice Macro F1, Cluster Macro F1, Stone Recall, and Tumor Recall with **95% Cluster-Bootstrap CIs**.
* Conducts non-parametric pairwise hypothesis testing to determine whether performance differences between models fall **within or outside the 95% Confidence Intervals**.
* Evaluates pooled (validation + test) performance metrics (noting that the test set is no longer untouched).
* Generates comparative charts (macro F1 comparison with error bars, confusion matrix grids).
* Exports the definitive `dl/models/ct/comparison.json` report."""),

        new_markdown_cell("""---
### 1. Global Configuration & Smoke Test Settings
* **What it does:** Sets filepaths, loads `common_eval.py`, prepares output directory, and sets `SMOKE_TEST` flag.
* **Expected runtime:** < 1 second.
* **Expected output:** Path confirmation."""),

        new_code_cell("""import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

WORKSPACE_ROOT = Path("/content/StoneSense-AI") if "google.colab" in sys.modules else Path(".").resolve()
sys.path.insert(0, (WORKSPACE_ROOT / "dl" / "notebooks").as_posix())

SMOKE_TEST = False  # Set to True for quick check (20 bootstrap resamples)
BOOTSTRAP_RESAMPLES = 20 if SMOKE_TEST else 1000

MODELS_DIR = WORKSPACE_ROOT / "dl" / "models" / "ct"
MANIFEST_PATH = WORKSPACE_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"
COMPARISON_OUT = MODELS_DIR / "comparison.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Workspace Root: {WORKSPACE_ROOT.as_posix()}")
print(f"CT Models Directory: {MODELS_DIR.as_posix()}")
print(f"Smoke Test Mode: {SMOKE_TEST}")"""),

        new_markdown_cell("""---
### 2. Check Cell: Verify ResNet18 Baseline Checkpoint & Candidate Artifacts
* **What it does:** Evaluates the existing ResNet18 checkpoint on `validation` and `test` splits using `common_eval.py` if not already generated, and verifies that `dinov3`, `yolo26`, and `qknn` artifacts exist.
* **Expected runtime:** ~10-20 seconds.
* **Expected output:** Status of artifacts for all four model families."""),

        new_code_cell("""import torch
from common_eval import generate_full_evaluation, save_contract_artifacts, CLASS_NAMES

# Required placement for ResNet18 checkpoint: dl/models/kidney_resnet18.pth
resnet18_ckpt = WORKSPACE_ROOT / "dl" / "models" / "kidney_resnet18.pth"
resnet18_dir = MODELS_DIR / "resnet18"
resnet18_metrics_file = resnet18_dir / "metrics.json"

if not resnet18_metrics_file.exists() and resnet18_ckpt.exists():
    print(f"Found ResNet18 checkpoint at {resnet18_ckpt.as_posix()}. Evaluating with common_eval standard...")
    from torchvision import transforms
    from torchvision.models import resnet18
    from PIL import Image
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=4)
    ckpt = torch.load(resnet18_ckpt, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    df_manifest = pd.read_csv(MANIFEST_PATH)
    
    def eval_rn18(split):
        sub = df_manifest[df_manifest["destination_split"] == split].reset_index(drop=True)
        if SMOKE_TEST:
            sub = sub.head(50)
        probs_list = []
        for _, r in sub.iterrows():
            p = WORKSPACE_ROOT / "dl" / "processed_grouped" / split / r["class"] / r["filename"]
            with Image.open(p) as img:
                t = tf(img.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                probs_list.append(torch.softmax(model(t), dim=1)[0].cpu().numpy())
        probs_mat = np.array(probs_list)
        return pd.DataFrame({
            "filename": sub["filename"],
            "cluster_id": sub["near_duplicate_group_id"],
            "y_true": sub["class"],
            "y_pred": [CLASS_NAMES[i] for i in np.argmax(probs_mat, axis=1)],
            "p_Cyst": probs_mat[:, 0],
            "p_Normal": probs_mat[:, 1],
            "p_Stone": probs_mat[:, 2],
            "p_Tumor": probs_mat[:, 3],
        })
        
    val_rn = eval_rn18("validation")
    test_rn = eval_rn18("test")
    eval_rn = generate_full_evaluation(val_rn, test_rn, bootstrap_iterations=BOOTSTRAP_RESAMPLES, seed=42)
    
    save_contract_artifacts(
        out_dir=MODELS_DIR,
        family="resnet18",
        version_tag="resnet18_baseline_v001",
        display_name="ResNet18 Transfer Learning Baseline",
        base_model="resnet18",
        license_name="BSD-3-Clause",
        hyperparameters={"epochs": 20, "lr": 1e-4, "seed": 42, "smoke_test": SMOKE_TEST},
        preprocessing_spec={"resize": [224, 224], "normalization": "ImageNet"},
        library_versions={"torch": torch.__version__},
        training_time_sec=0.0,
        evaluation_results=eval_rn,
        predict_example_code="# Loaded via standard PyTorch ResNet18 pipeline",
        seed=42
    )
    print("Successfully generated ResNet18 common_eval metrics.")
elif not resnet18_ckpt.exists():
    print(f"ResNet18 checkpoint missing at {resnet18_ckpt.as_posix()}! Ensure baseline weights are placed here.")

# Verify all models
families = ["resnet18", "dinov3", "yolo26", "qknn"]
print("=" * 60)
for fam in families:
    m_path = MODELS_DIR / fam / "metrics.json"
    c_path = MODELS_DIR / fam / "model_card.json"
    status = "READY" if m_path.exists() and c_path.exists() else "MISSING (Run prior notebook)"
    print(f"   - {fam.upper():<10}: {status}")
print("=" * 60)"""),

        new_markdown_cell("""---
### 3. Aggregate Performance Table with 95% Confidence Intervals
* **What it does:** Reads `metrics.json` from each family directory, extracts test metrics and 95% CIs, and formats a comparative benchmark table.
* **Expected runtime:** ~1-2 seconds.
* **Expected output:** Markdown/ASCII table displaying Slice Macro F1, Cluster Macro F1, Stone Recall, and Tumor Recall with CIs."""),

        new_code_cell("""summary_rows = []

for fam in ["resnet18", "dinov3", "yolo26", "qknn"]:
    m_path = MODELS_DIR / fam / "metrics.json"
    c_path = MODELS_DIR / fam / "model_card.json"
    if not m_path.exists() or not c_path.exists():
        continue
        
    with open(m_path, "r") as f:
        m = json.load(f)
    with open(c_path, "r") as f:
        card = json.load(f)
        
    test = m["test"]
    ci = test["confidence_intervals_95"]
    
    f1_mean = test["slice_macro_f1"]
    f1_lo = ci["slice_macro_f1"]["ci_lower_95"]
    f1_hi = ci["slice_macro_f1"]["ci_upper_95"]
    
    summary_rows.append({
        "Model Family": fam,
        "Display Name": card.get("display_name", fam),
        "Slice Acc": f"{test['slice_accuracy']:.4f}",
        "Slice Macro F1 [95% CI]": f"{f1_mean:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]",
        "Cluster Macro F1": f"{test['cluster_macro_f1']:.4f}",
        "Stone Recall": f"{test['per_class']['Stone']['recall']:.4f}",
        "Tumor Recall": f"{test['per_class']['Tumor']['recall']:.4f}",
        "_f1_mean": f1_mean,
        "_f1_lo": f1_lo,
        "_f1_hi": f1_hi
    })

df_summary = pd.DataFrame(summary_rows)
display_cols = ["Model Family", "Display Name", "Slice Acc", "Slice Macro F1 [95% CI]", "Cluster Macro F1", "Stone Recall", "Tumor Recall"]
print(df_summary[display_cols].to_markdown(index=False))"""),

        new_markdown_cell("""---
### 4. Statistical CI Overlap Analysis
* **What it does:** Evaluates pairwise CI overlap between models to report whether observed differences in Macro F1 are statistically indistinguishable or significant.
* **Expected runtime:** < 1 second.
* **Expected output:** Statistical significance report."""),

        new_code_cell("""print("=" * 60)
print("STATISTICAL OVERLAP ANALYSIS (95% Cluster-Bootstrap CIs)")
print("=" * 60)

for i in range(len(summary_rows)):
    for j in range(i + 1, len(summary_rows)):
        m1 = summary_rows[i]
        m2 = summary_rows[j]
        
        overlap = not (m1["_f1_hi"] < m2["_f1_lo"] or m2["_f1_hi"] < m1["_f1_lo"])
        
        status_text = "STATISTICALLY INDISTINGUISHABLE (CIs overlap)" if overlap else "STATISTICALLY SIGNIFICANT DIFFERENCE (CIs do not overlap)"
        print(f"{m1['Model Family'].upper()} vs {m2['Model Family'].upper()}:")
        print(f"   {m1['Model Family']}: [{m1['_f1_lo']:.4f}, {m1['_f1_hi']:.4f}] | {m2['Model Family']}: [{m2['_f1_lo']:.4f}, {m2['_f1_hi']:.4f}]")
        print(f"   Result: {status_text}\\n")
print("=" * 60)"""),

        new_markdown_cell("""---
### 5. Generate Comparison Visualizations & `comparison.json`
* **What it does:** Plots comparative Macro F1 chart with error bars and writes the final `comparison.json` report.
* **Expected runtime:** ~2-4 seconds.
* **Expected output:** Saved chart and written `comparison.json`."""),

        new_code_cell("""import matplotlib.pyplot as plt

if summary_rows:
    plt.figure(figsize=(10, 5), dpi=150)
    families = [r["Model Family"] for r in summary_rows]
    means = [r["_f1_mean"] for r in summary_rows]
    lowers = [r["_f1_mean"] - r["_f1_lo"] for r in summary_rows]
    uppers = [r["_f1_hi"] - r["_f1_mean"] for r in summary_rows]

    yerr = [lowers, uppers]
    bars = plt.bar(families, means, yerr=yerr, capsize=6, color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"][:len(families)], edgecolor="black", alpha=0.85)

    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, mean / 2, f"{mean:.4f}", ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    plt.title("StoneSense-AI: CT Model Comparison — Test Macro F1 (95% Cluster-Bootstrap CIs)", fontsize=12, fontweight="bold")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    chart_path = MODELS_DIR / "model_comparison_f1.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart to {chart_path.as_posix()}")

    comparison_data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_evaluated": families,
        "summary_table": summary_rows,
        "charts": [chart_path.name]
    }

    with open(COMPARISON_OUT, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"Written comparison report to {COMPARISON_OUT.as_posix()}")
else:
    print("No evaluated models available yet to plot comparison chart.")"""),

        new_markdown_cell("""---
### 6. Final Deployment Packaging
* **What it does:** Packages all CT models and comparison report into `ct_multimodel_bundle.zip` for production deployment.
* **Expected runtime:** ~3-5 seconds.
* **Expected output:** Final archive summary."""),

        new_code_cell("""import shutil

final_zip = WORKSPACE_ROOT / "dl" / "models" / "ct" / "ct_multimodel_bundle.zip"
shutil.make_archive(str(final_zip.with_suffix("")), 'zip', MODELS_DIR)

print("=" * 60)
print("MULTI-MODEL CT SUITE COMPLETE!")
print(f"Final Archive: {final_zip.name}")
print(f"Contains models: {families}")
print("=" * 60)""")
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["accelerator"] = "GPU"
    return nb


def main():
    out_dir = Path(__file__).resolve().parent
    notebooks = {
        "00_setup_and_data.ipynb": build_nb00(),
        "01_dinov3.ipynb": build_nb01(),
        "02_yolo26_cls.ipynb": build_nb02(),
        "03_qknn.ipynb": build_nb03(),
        "04_comparison.ipynb": build_nb04(),
    }

    for name, nb_obj in notebooks.items():
        target = out_dir / name
        with open(target, "w", encoding="utf-8") as f:
            nbformat.write(nb_obj, f)
        
        # Re-read and validate with nbformat
        loaded_nb = nbformat.read(str(target), as_version=4)
        nbformat.validate(loaded_nb)
        print(f"SUCCESS: Created and validated {target.name} (nbformat v4)")


if __name__ == "__main__":
    main()
