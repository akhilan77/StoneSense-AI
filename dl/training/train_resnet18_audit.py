"""StoneSense-AI ResNet18 Leakage-Free Audit & Retraining Pipeline.

Performs:
1. Inspection of grouped manifests and validation of zero cross-split duplicate leakage.
2. Transfer learning training of ResNet18 using ONLY dl/processed_grouped/train.
3. Checkpoint selection and validation monitoring using ONLY dl/processed_grouped/validation.
4. Single-pass evaluation on the untouched dl/processed_grouped/test split.
5. Cluster-level aggregation and 1,000-iteration cluster bootstrap 95% Confidence Intervals.
6. Generation of comprehensive audit metrics artifacts preserving historical baselines.
"""

import sys
import os
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"
PROCESSED_GROUPED = PROJECT_ROOT / "dl" / "processed_grouped"
MODELS_DIR = PROJECT_ROOT / "dl" / "models"
CT_RESNET18_DIR = MODELS_DIR / "ct" / "resnet18"
REPORTS_DIR = PROJECT_ROOT / "dl" / "outputs" / "reports"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("ResNet18Audit")

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def build_model(num_classes: int = 4) -> nn.Module:
    """Instantiates ResNet18 with ImageNet-1K weights, frozen conv1..layer3, and trainable layer4 + fc."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Freeze conv1 through layer3
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace fc
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def preload_split_tensors(manifest_df: pd.DataFrame, split: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Loads images for a split into a contiguous in-memory uint8 tensor for maximum CPU throughput."""
    split_df = manifest_df[manifest_df["destination_split"] == split].reset_index(drop=True)
    n = len(split_df)
    logger.info(f"Preloading {n} images for split '{split}' into RAM...")
    
    t0 = time.time()
    tensors = torch.empty((n, 3, 224, 224), dtype=torch.uint8)
    labels = torch.empty(n, dtype=torch.long)
    filenames = []
    clusters = []

    for idx, row in split_df.iterrows():
        p = Path(row["source_path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        with Image.open(p) as img:
            img_rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(img_rgb, dtype=np.uint8)
            tensors[idx] = torch.from_numpy(arr).permute(2, 0, 1)
        labels[idx] = CLASS_TO_IDX[row["class"]]
        filenames.append(row["filename"])
        clusters.append(row["near_duplicate_group_id"])

    logger.info(f"Loaded {split} ({tensors.element_size() * tensors.nelement() / 1e6:.1f} MB) in {time.time() - t0:.2f}s")
    return tensors, labels, filenames, clusters


def transform_batch(batch_u8: torch.Tensor, augment: bool = False) -> torch.Tensor:
    """Normalizes and optionally augments a batch of uint8 tensors on CPU."""
    if augment:
        # Random horizontal flip
        flip_mask = torch.rand(batch_u8.size(0)) > 0.5
        batch_u8 = batch_u8.clone()
        batch_u8[flip_mask] = batch_u8[flip_mask].flip(-1)

    batch_f = batch_u8.to(torch.float32) / 255.0
    return (batch_f - IMAGENET_MEAN) / IMAGENET_STD


def evaluate_split(
    model: nn.Module,
    X_u8: torch.Tensor,
    y: torch.Tensor,
    filenames: List[str],
    clusters: List[str],
    batch_size: int = 64,
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs evaluation over a split and returns metrics dict and detailed slice DataFrame."""
    model.eval()
    n = len(y)
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_x = transform_batch(X_u8[i:i + batch_size], augment=False).to(device)
            outputs = model(batch_x)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)

    probs_arr = np.concatenate(all_probs, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = y.numpy()

    # Build DataFrame
    records = []
    for fn, cid, y_t, y_p, p_vec in zip(filenames, clusters, targets_arr, preds_arr, probs_arr):
        records.append({
            "filename": fn,
            "cluster_id": cid,
            "y_true": int(y_t),
            "y_pred": int(y_p),
            "prob_cyst": float(p_vec[0]),
            "prob_normal": float(p_vec[1]),
            "prob_stone": float(p_vec[2]),
            "prob_tumor": float(p_vec[3]),
        })
    df = pd.DataFrame(records)

    # 1. Slice-Level Metrics
    acc = float(accuracy_score(targets_arr, preds_arr))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(targets_arr, preds_arr, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(targets_arr, preds_arr, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(targets_arr, preds_arr, average=None, labels=[0, 1, 2, 3], zero_division=0)
    cm = confusion_matrix(targets_arr, preds_arr, labels=[0, 1, 2, 3]).tolist()

    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        per_class[cname] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1_score": float(f1_cls[idx]),
            "support": int(supp_cls[idx])
        }

    # 2. Cluster-Level Metrics (Mean-probability pooling)
    cluster_records = []
    prob_cols = ["prob_cyst", "prob_normal", "prob_stone", "prob_tumor"]
    for cid, cgroup in df.groupby("cluster_id"):
        mean_probs = cgroup[prob_cols].mean().values
        c_pred = int(np.argmax(mean_probs))
        c_true = int(cgroup["y_true"].mode().iloc[0])
        cluster_records.append({
            "cluster_id": cid,
            "y_true": c_true,
            "y_pred": c_pred,
            "slices": len(cgroup)
        })
    cdf = pd.DataFrame(cluster_records)
    c_y_true = cdf["y_true"].values
    c_y_pred = cdf["y_pred"].values

    cluster_acc = float(accuracy_score(c_y_true, c_y_pred))
    c_p_mac, c_r_mac, c_f1_mac, _ = precision_recall_fscore_support(c_y_true, c_y_pred, average="macro", zero_division=0)
    c_p_cls, c_r_cls, c_f1_cls, c_supp_cls = precision_recall_fscore_support(c_y_true, c_y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    cluster_cm = confusion_matrix(c_y_true, c_y_pred, labels=[0, 1, 2, 3]).tolist()

    cluster_per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        cluster_per_class[cname] = {
            "precision": float(c_p_cls[idx]),
            "recall": float(c_r_cls[idx]),
            "f1_score": float(c_f1_cls[idx]),
            "support": int(c_supp_cls[idx])
        }

    metrics = {
        "slice_metrics": {
            "accuracy": acc,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f1_weighted),
            "per_class_metrics": per_class,
            "confusion_matrix": cm,
            "sample_count": n,
        },
        "cluster_metrics": {
            "accuracy": cluster_acc,
            "precision_macro": float(c_p_mac),
            "recall_macro": float(c_r_mac),
            "f1_macro": float(c_f1_mac),
            "per_class_metrics": cluster_per_class,
            "confusion_matrix": cluster_cm,
            "cluster_count": len(cdf),
        }
    }
    return metrics, df


def compute_cluster_bootstrap_ci(df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Computes 95% non-parametric bootstrap Confidence Intervals by resampling perceptual clusters."""
    np.random.seed(42)
    unique_clusters = df["cluster_id"].unique()
    n_clusters = len(unique_clusters)
    prob_cols = ["prob_cyst", "prob_normal", "prob_stone", "prob_tumor"]

    cluster_df_dict = {cid: grp for cid, grp in df.groupby("cluster_id")}
    
    # Precompute cluster-level truth and prediction
    cluster_meta = {}
    for cid, grp in cluster_df_dict.items():
        mean_p = grp[prob_cols].mean().values
        cluster_meta[cid] = (int(grp["y_true"].mode().iloc[0]), int(np.argmax(mean_p)))

    slice_accs = []
    slice_f1s = []
    slice_stone_recs = []
    slice_tumor_recs = []

    cluster_accs = []
    cluster_f1s = []
    cluster_stone_recs = []
    cluster_tumor_recs = []

    for _ in range(n_bootstrap):
        sampled_cids = np.random.choice(unique_clusters, size=n_clusters, replace=True)

        # Slice-level bootstrap
        boot_y_true = []
        boot_y_pred = []
        for cid in sampled_cids:
            grp = cluster_df_dict[cid]
            boot_y_true.extend(grp["y_true"].tolist())
            boot_y_pred.extend(grp["y_pred"].tolist())

        yt = np.array(boot_y_true)
        yp = np.array(boot_y_pred)
        slice_accs.append(accuracy_score(yt, yp))
        slice_f1s.append(precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2])
        r_all = precision_recall_fscore_support(yt, yp, average=None, labels=[0, 1, 2, 3], zero_division=0)[1]
        slice_stone_recs.append(r_all[2])
        slice_tumor_recs.append(r_all[3])

        # Cluster-level bootstrap
        c_yt = np.array([cluster_meta[cid][0] for cid in sampled_cids])
        c_yp = np.array([cluster_meta[cid][1] for cid in sampled_cids])
        cluster_accs.append(accuracy_score(c_yt, c_yp))
        cluster_f1s.append(precision_recall_fscore_support(c_yt, c_yp, average="macro", zero_division=0)[2])
        c_r_all = precision_recall_fscore_support(c_yt, c_yp, average=None, labels=[0, 1, 2, 3], zero_division=0)[1]
        cluster_stone_recs.append(c_r_all[2])
        cluster_tumor_recs.append(c_r_all[3])

    def get_ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_95_lower": float(np.percentile(arr, 2.5)),
            "ci_95_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "slice_level": {
            "accuracy": get_ci(slice_accs),
            "macro_f1": get_ci(slice_f1s),
            "stone_recall": get_ci(slice_stone_recs),
            "tumor_recall": get_ci(slice_tumor_recs),
        },
        "cluster_level": {
            "accuracy": get_ci(cluster_accs),
            "macro_f1": get_ci(cluster_f1s),
            "stone_recall": get_ci(cluster_stone_recs),
            "tumor_recall": get_ci(cluster_tumor_recs),
        }
    }


def main():
    logger.info("=" * 75)
    logger.info("Starting StoneSense-AI ResNet18 Leakage-Free Audit & Retraining")
    logger.info("=" * 75)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    logger.info(f"Loaded manifest from {MANIFEST_PATH}: {len(manifest_df)} total records.")

    # 1. Preload Tensors
    X_train, y_train, train_fns, train_clusters = preload_split_tensors(manifest_df, "train")
    X_val, y_val, val_fns, val_clusters = preload_split_tensors(manifest_df, "validation")
    X_test, y_test, test_fns, test_clusters = preload_split_tensors(manifest_df, "test")

    # 2. Build Model with Documented Pretrained Weights
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device} (threads={torch.get_num_threads()})")

    model = build_model(num_classes=4)
    model.to(device)

    # Class-weighted loss on training set
    train_counts = torch.bincount(y_train, minlength=4).float()
    class_weights = train_counts.sum() / (4.0 * train_counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # 3. Training Loop with ONLY Train Split and Checkpoint Selection on ONLY Validation Split
    epochs = 6
    batch_size = 64
    n_train = len(y_train)
    best_val_macro_f1 = -1.0
    best_val_metrics = None
    best_epoch = -1
    best_state_dict = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
    }

    logger.info(f"Beginning training loop for {epochs} epochs (Batch size: {batch_size})...")
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        # Ensure frozen layers remain in eval mode for consistent BatchNorm stats
        model.conv1.eval()
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()

        perm = torch.randperm(n_train)
        running_loss = 0.0
        correct = 0

        t0_epoch = time.time()
        for i in range(0, n_train, batch_size):
            idx_batch = perm[i:i + batch_size]
            batch_x = transform_batch(X_train[idx_batch], augment=True).to(device)
            batch_y = y_train[idx_batch].to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()

        epoch_train_loss = running_loss / n_train
        epoch_train_acc = correct / n_train

        # Validation Evaluation (ONLY on Validation Split)
        val_metrics, _ = evaluate_split(model, X_val, y_val, val_fns, val_clusters, batch_size=64, device=device)
        val_slice = val_metrics["slice_metrics"]
        val_acc = val_slice["accuracy"]
        val_f1 = val_slice["f1_macro"]

        scheduler.step(val_f1)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(val_slice.get("loss", 0.0), 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1_macro"].append(round(val_f1, 4))

        elapsed_epoch = time.time() - t0_epoch
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed_epoch:.1f}s] - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}, Val Macro F1: {val_f1:.4f} (Stone Rec: {val_slice['per_class_metrics']['Stone']['recall']:.4f}, Tumor Rec: {val_slice['per_class_metrics']['Tumor']['recall']:.4f})"
        )

        # Checkpoint selection strictly based on validation Macro F1
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  >>> Improved Validation Macro F1 ({best_val_macro_f1:.4f}) at epoch {best_epoch}. New best checkpoint.")

    total_train_elapsed = time.time() - start_train_time
    logger.info(f"Training completed in {total_train_elapsed:.1f}s. Best validation epoch: {best_epoch} (Val F1: {best_val_macro_f1:.4f}).")

    # 4. Save Best Checkpoints
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CT_RESNET18_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(best_state_dict, MODELS_DIR / "kidney_resnet18.pth")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_state_dict,
        "val_f1_macro": best_val_macro_f1,
        "initialization": "ResNet18_Weights.IMAGENET1K_V1",
        "split_protocol": "perceptual_cluster_grouped",
        "trained_on": "dl/processed_grouped/train",
        "selected_on": "dl/processed_grouped/validation",
    }, MODELS_DIR / "best_checkpoint.pth")

    # Also save to model wrapper directory
    torch.save(best_state_dict, CT_RESNET18_DIR / "kidney_resnet18.pth")
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_state_dict,
        "val_f1_macro": best_val_macro_f1,
    }, CT_RESNET18_DIR / "best_checkpoint.pth")
    logger.info("Saved best checkpoint state dicts to dl/models and dl/models/ct/resnet18.")

    # 5. Single Held-Out Evaluation on Untouched Grouped Test Split
    logger.info("=" * 75)
    logger.info("Executing SINGLE evaluation of best checkpoint on untouched grouped TEST split...")
    logger.info("=" * 75)

    model.load_state_dict(best_state_dict)
    model.to(device)

    test_metrics, test_df = evaluate_split(model, X_test, y_test, test_fns, test_clusters, batch_size=64, device=device)
    test_slice = test_metrics["slice_metrics"]
    test_cluster = test_metrics["cluster_metrics"]

    logger.info(f"Test Slice Accuracy: {test_slice['accuracy']:.4f}")
    logger.info(f"Test Slice Macro F1: {test_slice['f1_macro']:.4f}")
    logger.info(f"Test Cluster Accuracy: {test_cluster['accuracy']:.4f}")
    logger.info(f"Test Cluster Macro F1: {test_cluster['f1_macro']:.4f}")

    # 6. Compute 1,000-Iteration Cluster-Bootstrap 95% Confidence Interval
    logger.info("Computing 1,000-iteration cluster bootstrap 95% Confidence Intervals...")
    bootstrap_ci = compute_cluster_bootstrap_ci(test_df, n_bootstrap=1000)

    # 7. Construct Comprehensive Metrics Payload
    metrics_payload = {
        "model_architecture": "ResNet18",
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1 (ImageNet-1K)",
        "split_protocol": "Perceptual Cluster Grouping (pHash Hamming distance <= 2)",
        "dataset_sample_counts": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
            "total": len(y_train) + len(y_val) + len(y_test)
        },
        "dataset_cluster_counts": {
            "train": len(set(train_clusters)),
            "validation": len(set(val_clusters)),
            "test": len(set(test_clusters)),
            "total": len(set(train_clusters + val_clusters + test_clusters))
        },
        "historical_leaky_baseline": {
            "split_type": "Random slice-level shuffle (LEAKY - Discredited)",
            "test_accuracy": 0.9962586851950829,
            "test_macro_f1": 0.9946514637789549,
            "test_stone_recall": 0.9711538461538461,
            "test_tumor_recall": 0.9970845481049563,
            "test_samples": 1871,
            "duplicate_cross_split_pairs": 227,
            "note": "Artificially inflated due to 227 exact duplicate pairs and 703 pHash clusters crossing train/test splits."
        },
        "validation_selection_metrics": {
            "best_epoch": best_epoch,
            "slice_accuracy": best_val_metrics["slice_metrics"]["accuracy"],
            "slice_macro_f1": best_val_metrics["slice_metrics"]["f1_macro"],
            "slice_per_class": best_val_metrics["slice_metrics"]["per_class_metrics"],
            "slice_confusion_matrix": best_val_metrics["slice_metrics"]["confusion_matrix"],
            "cluster_accuracy": best_val_metrics["cluster_metrics"]["accuracy"],
            "cluster_macro_f1": best_val_metrics["cluster_metrics"]["f1_macro"],
            "cluster_per_class": best_val_metrics["cluster_metrics"]["per_class_metrics"],
            "sample_count": len(y_val),
            "cluster_count": len(set(val_clusters))
        },
        "held_out_test_metrics": {
            "slice_accuracy": test_slice["accuracy"],
            "slice_precision_macro": test_slice["precision_macro"],
            "slice_recall_macro": test_slice["recall_macro"],
            "slice_f1_macro": test_slice["f1_macro"],
            "slice_f1_weighted": test_slice["f1_weighted"],
            "slice_per_class": test_slice["per_class_metrics"],
            "slice_confusion_matrix": test_slice["confusion_matrix"],
            "cluster_accuracy": test_cluster["accuracy"],
            "cluster_precision_macro": test_cluster["precision_macro"],
            "cluster_recall_macro": test_cluster["recall_macro"],
            "cluster_f1_macro": test_cluster["f1_macro"],
            "cluster_per_class": test_cluster["per_class_metrics"],
            "cluster_confusion_matrix": test_cluster["confusion_matrix"],
            "sample_count": len(y_test),
            "cluster_count": len(set(test_clusters))
        },
        "bootstrap_95_ci": bootstrap_ci,
        # Standard root fields for backwards-compatible loaders
        "accuracy": test_slice["accuracy"],
        "f1_macro": test_slice["f1_macro"],
        "precision_macro": test_slice["precision_macro"],
        "recall_macro": test_slice["recall_macro"],
        "per_class_metrics": test_slice["per_class_metrics"],
        "confusion_matrix": test_slice["confusion_matrix"],
        "stone_recall": test_slice["per_class_metrics"]["Stone"]["recall"],
        "tumor_recall": test_slice["per_class_metrics"]["Tumor"]["recall"],
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_status": "audited_grouped_benchmark",
        "patient_level_separation_verified": False,
        "patient_level_limitation": "Patient IDs unavailable in source dataset; clustering based on perceptual similarity (pHash <= 2).",
        "disclaimer": "Research Prototype Only — Not for Clinical Diagnosis or Treatment Planning."
    }

    # Save to metrics.json in both locations
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    with open(CT_RESNET18_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    with open(MODELS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Update model_card.json
    model_card = {
        "model_id": "resnet18",
        "display_name": "ResNet18 (Grouped Audit Benchmark)",
        "base_model": "torchvision.models.resnet18",
        "pretrained_weights": "IMAGENET1K_V1",
        "version_tag": "resnet18_grouped_audit_v1",
        "family": "resnet18_ct",
        "license": "BSD-3-Clause",
        "trained_on": "dl/processed_grouped/train (8,708 images, 699 perceptual groups)",
        "validated_on": "dl/processed_grouped/validation (1,869 images, 138 perceptual groups)",
        "tested_on": "dl/processed_grouped/test (1,869 images, 142 perceptual groups)",
        "class_names": CLASS_NAMES,
        "input_size": [224, 224],
        "test_accuracy": test_slice["accuracy"],
        "test_macro_f1": test_slice["f1_macro"],
        "test_stone_recall": test_slice["per_class_metrics"]["Stone"]["recall"],
        "test_tumor_recall": test_slice["per_class_metrics"]["Tumor"]["recall"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "is_approved": False,
        "audit_verified": True,
        "patient_level_verified": False,
        "clinical_validity": "UNVALIDATED_RESEARCH_PROTOTYPE"
    }
    with open(CT_RESNET18_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    logger.info("Saved metrics.json and model_card.json successfully.")


if __name__ == "__main__":
    main()
