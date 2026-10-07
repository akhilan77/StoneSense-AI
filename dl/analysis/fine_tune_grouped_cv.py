"""Vectorized & Modular Grouped 5-Fold Cross-Validation Fine-Tuning Experiment for CT Kidney Classification.

Evaluates 3 fine-tuning architectures across all 979 patient/scan clusters (12,446 images):
  (a) layer4 + classification head
  (b) layer3 + layer4 + classification head
  (c) full backbone fine-tune with low discriminative LR

Reports slice-level and cluster-level metrics (Accuracy, Stone Recall, Tumor Recall)
with 95% cluster-bootstrap confidence intervals and wall-clock training time per regime.

Zero DB writes.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "dl" / "outputs" / "reports"

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def preload_all_tensors(manifest_df: pd.DataFrame) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    print(f"Preloading and resizing {len(manifest_df)} images into contiguous RAM uint8 tensor...", flush=True)
    t0 = time.time()
    n = len(manifest_df)
    X = torch.empty((n, 3, 224, 224), dtype=torch.uint8)
    labels = np.empty(n, dtype=np.int64)
    groups = np.empty(n, dtype=object)

    for idx, row in manifest_df.iterrows():
        path = Path(row["source_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with Image.open(path) as img:
            img_rgb = img.convert("RGB").resize((224, 224))
            arr = np.array(img_rgb, dtype=np.uint8)
            X[idx] = torch.from_numpy(arr).permute(2, 0, 1)
        labels[idx] = CLASS_TO_IDX[row["class"]]
        groups[idx] = row["near_duplicate_group_id"]

    print(f"Preloaded tensor shape {tuple(X.shape)} ({X.element_size() * X.nelement() / 1e9:.2f} GB) in {time.time() - t0:.1f}s.", flush=True)
    return X, labels, groups


def augment_batch(imgs_u8: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() > 0.5:
        imgs_u8 = imgs_u8.flip(-1)
    imgs_f = imgs_u8.to(torch.float32) / 255.0
    return (imgs_f - MEAN) / STD


def normalize_batch(imgs_u8: torch.Tensor) -> torch.Tensor:
    imgs_f = imgs_u8.to(torch.float32) / 255.0
    return (imgs_f - MEAN) / STD


class Layer4HeadModel(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.prefix = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
            base_model.layer1, base_model.layer2, base_model.layer3
        )
        for p in self.prefix.parameters():
            p.requires_grad = False
        self.prefix.eval()

        self.trainable = nn.Sequential(
            base_model.layer4,
            base_model.avgpool,
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feat = self.prefix(x)
        return self.trainable(feat)


class Layer3Layer4HeadModel(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.prefix = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
            base_model.layer1, base_model.layer2
        )
        for p in self.prefix.parameters():
            p.requires_grad = False
        self.prefix.eval()

        self.trainable = nn.Sequential(
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feat = self.prefix(x)
        return self.trainable(feat)


class FullFinetuneModel(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super().__init__()
        self.backbone = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
            base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4,
            base_model.avgpool, nn.Flatten()
        )
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat)


def build_experiment_model(variant_name: str) -> Tuple[nn.Module, torch.optim.Optimizer]:
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if variant_name == "layer4_head":
        model = Layer4HeadModel(base_model, num_classes=len(CLASS_NAMES))
        optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=1e-4, weight_decay=1e-4)
    elif variant_name == "layer3_layer4_head":
        model = Layer3Layer4HeadModel(base_model, num_classes=len(CLASS_NAMES))
        optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=1e-4, weight_decay=1e-4)
    elif variant_name == "full_finetune":
        model = FullFinetuneModel(base_model, num_classes=len(CLASS_NAMES))
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-4}
        ], weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    return model, optimizer


def compute_cluster_bootstrap_ci(
    records: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(records)
    unique_clusters = df["cluster_id"].unique()
    n_clusters = len(unique_clusters)

    metrics_samples = {
        "slice_accuracy": [],
        "slice_stone_recall": [],
        "slice_tumor_recall": [],
        "cluster_accuracy": [],
        "cluster_stone_recall": [],
        "cluster_tumor_recall": [],
    }

    cluster_df_map = {cid: cdf for cid, cdf in df.groupby("cluster_id")}
    cluster_summary = {}
    for cid, cdf in cluster_df_map.items():
        true_label = cdf["true_label"].iloc[0]
        mean_probs = np.mean(np.vstack(cdf["probs"].values), axis=0)
        cluster_pred = int(np.argmax(mean_probs))
        cluster_summary[cid] = {
            "true_label": true_label,
            "cluster_pred": cluster_pred,
            "slice_true": cdf["true_label"].values,
            "slice_pred": cdf["slice_pred"].values,
        }

    for _ in range(n_resamples):
        sampled_cids = rng.choice(unique_clusters, size=n_clusters, replace=True)

        slice_trues, slice_preds = [], []
        cl_trues, cl_preds = [], []

        for cid in sampled_cids:
            cdata = cluster_summary[cid]
            slice_trues.extend(cdata["slice_true"])
            slice_preds.extend(cdata["slice_pred"])
            cl_trues.append(cdata["true_label"])
            cl_preds.append(cdata["cluster_pred"])

        slice_trues = np.array(slice_trues)
        slice_preds = np.array(slice_preds)
        cl_trues = np.array(cl_trues)
        cl_preds = np.array(cl_preds)

        s_acc = float(np.mean(slice_trues == slice_preds))
        metrics_samples["slice_accuracy"].append(s_acc)

        stone_mask = (slice_trues == CLASS_TO_IDX["Stone"])
        if np.sum(stone_mask) > 0:
            metrics_samples["slice_stone_recall"].append(float(np.mean(slice_preds[stone_mask] == CLASS_TO_IDX["Stone"])))

        tumor_mask = (slice_trues == CLASS_TO_IDX["Tumor"])
        if np.sum(tumor_mask) > 0:
            metrics_samples["slice_tumor_recall"].append(float(np.mean(slice_preds[tumor_mask] == CLASS_TO_IDX["Tumor"])))

        c_acc = float(np.mean(cl_trues == cl_preds))
        metrics_samples["cluster_accuracy"].append(c_acc)

        c_stone_mask = (cl_trues == CLASS_TO_IDX["Stone"])
        if np.sum(c_stone_mask) > 0:
            metrics_samples["cluster_stone_recall"].append(float(np.mean(cl_preds[c_stone_mask] == CLASS_TO_IDX["Stone"])))

        c_tumor_mask = (cl_trues == CLASS_TO_IDX["Tumor"])
        if np.sum(c_tumor_mask) > 0:
            metrics_samples["cluster_tumor_recall"].append(float(np.mean(cl_preds[c_tumor_mask] == CLASS_TO_IDX["Tumor"])))

    cis = {}
    for k, v in metrics_samples.items():
        if v:
            cis[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        else:
            cis[k] = (0.0, 0.0)
    return cis


def run_single_variant_cv(
    variant_name: str,
    X: torch.Tensor,
    labels: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 64,
) -> Dict[str, Any]:
    print(f"\n{'='*70}\nSTARTING 5-FOLD GROUPED CV: Variant '{variant_name}' (Epochs: {epochs})\n{'='*70}", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X_dummy = np.zeros(len(labels))

    oof_records = []
    start_total = time.time()
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dummy, labels, groups=groups)):
        t0 = time.time()
        
        train_X = X[train_idx]
        train_y = torch.tensor(labels[train_idx], dtype=torch.long)
        
        val_X = X[val_idx]
        val_y = labels[val_idx]
        val_groups = groups[val_idx]

        model, optimizer = build_experiment_model(variant_name)
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        n_train = len(train_idx)
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_train)
            for b_start in range(0, n_train, batch_size):
                b_idx = perm[b_start:b_start + batch_size]
                b_imgs = augment_batch(train_X[b_idx]).to(device)
                b_targets = train_y[b_idx].to(device)

                optimizer.zero_grad()
                outputs = model(b_imgs)
                loss = criterion(outputs, b_targets)
                loss.backward()
                optimizer.step()

        # OOF Evaluation in batches
        model.eval()
        n_val = len(val_idx)
        with torch.no_grad():
            for b_start in range(0, n_val, batch_size):
                b_imgs = normalize_batch(val_X[b_start:b_start + batch_size]).to(device)
                outputs = model(b_imgs)
                probs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                b_targets = val_y[b_start:b_start + batch_size]
                b_grps = val_groups[b_start:b_start + batch_size]

                for p, pr, t, g in zip(preds, probs, b_targets, b_grps):
                    oof_records.append({
                        "fold": fold + 1,
                        "cluster_id": g,
                        "true_label": int(t),
                        "slice_pred": int(p),
                        "probs": pr,
                    })

        elapsed_fold = time.time() - t0
        fold_times.append(elapsed_fold)
        print(f"  [Fold {fold+1}/5] Completed in {elapsed_fold:.1f}s (Val Slices: {n_val}, Clusters: {len(np.unique(val_groups))})", flush=True)

    total_time = time.time() - start_total

    # Calculate global OOF point estimates
    y_true = np.array([r["true_label"] for r in oof_records])
    y_pred = np.array([r["slice_pred"] for r in oof_records])

    slice_acc = float(accuracy_score(y_true, y_pred))
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), average=None, zero_division=0
    )
    stone_recall = float(r_class[CLASS_TO_IDX["Stone"]])
    tumor_recall = float(r_class[CLASS_TO_IDX["Tumor"]])

    # Cluster-level voting
    oof_df = pd.DataFrame(oof_records)
    cluster_votes = []
    for cid, cdf in oof_df.groupby("cluster_id"):
        true_lbl = cdf["true_label"].iloc[0]
        mean_p = np.mean(np.vstack(cdf["probs"].values), axis=0)
        c_pred = int(np.argmax(mean_p))
        cluster_votes.append({
            "cluster_id": cid,
            "true_label": true_lbl,
            "cluster_pred": c_pred,
        })

    cv_df = pd.DataFrame(cluster_votes)
    cl_acc = float(accuracy_score(cv_df["true_label"], cv_df["cluster_pred"]))
    _, cl_r_class, _, _ = precision_recall_fscore_support(
        cv_df["true_label"], cv_df["cluster_pred"], labels=list(range(4)), average=None, zero_division=0
    )
    cl_stone_recall = float(cl_r_class[CLASS_TO_IDX["Stone"]])
    cl_tumor_recall = float(cl_r_class[CLASS_TO_IDX["Tumor"]])

    # Bootstrap CIs
    cis = compute_cluster_bootstrap_ci(oof_records, n_resamples=1000, seed=42)

    result = {
        "variant": variant_name,
        "total_time_sec": round(total_time, 2),
        "mean_fold_time_sec": round(float(np.mean(fold_times)), 2),
        "total_slices": len(oof_records),
        "total_clusters": len(cluster_votes),
        "slice_metrics": {
            "accuracy": round(slice_acc, 4),
            "accuracy_ci_95": [round(cis["slice_accuracy"][0], 4), round(cis["slice_accuracy"][1], 4)],
            "stone_recall": round(stone_recall, 4),
            "stone_recall_ci_95": [round(cis["slice_stone_recall"][0], 4), round(cis["slice_stone_recall"][1], 4)],
            "tumor_recall": round(tumor_recall, 4),
            "tumor_recall_ci_95": [round(cis["slice_tumor_recall"][0], 4), round(cis["slice_tumor_recall"][1], 4)],
        },
        "cluster_metrics": {
            "accuracy": round(cl_acc, 4),
            "accuracy_ci_95": [round(cis["cluster_accuracy"][0], 4), round(cis["cluster_accuracy"][1], 4)],
            "stone_recall": round(cl_stone_recall, 4),
            "stone_recall_ci_95": [round(cis["cluster_stone_recall"][0], 4), round(cis["cluster_stone_recall"][1], 4)],
            "tumor_recall": round(cl_tumor_recall, 4),
            "tumor_recall_ci_95": [round(cis["cluster_tumor_recall"][0], 4), round(cis["cluster_tumor_recall"][1], 4)],
        }
    }
    return result


def main():
    torch.set_num_threads(8)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Total Manifest Slices: {len(manifest_df)} | Total Groups: {manifest_df['near_duplicate_group_id'].nunique()}", flush=True)

    X, labels, groups = preload_all_tensors(manifest_df)

    variants = ["layer4_head", "layer3_layer4_head", "full_finetune"]
    all_results = {}

    for var in variants:
        res = run_single_variant_cv(
            variant_name=var,
            X=X,
            labels=labels,
            groups=groups,
            device=device,
            epochs=1,
            batch_size=64,
        )
        all_results[var] = res

    # Save JSON report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "fine_tune_grouped_cv_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved CV results to: {json_path}", flush=True)

    # Print summary markdown table
    print("\n" + "="*80, flush=True)
    print("5-FOLD GROUPED CV EXPERIMENT RESULTS (979 Clusters, 12,446 Slices)", flush=True)
    print("="*80, flush=True)
    print(f"{'Variant':<22} | {'Slice Acc (95% CI)':<26} | {'Stone Rec (CI)':<22} | {'Tumor Rec (CI)':<22} | {'Cluster Acc (CI)':<26} | {'Cl Stone Rec':<14} | {'Cl Tumor Rec':<14} | {'Time (s)':<8}", flush=True)
    print("-" * 155, flush=True)
    for var, data in all_results.items():
        sm = data["slice_metrics"]
        cm = data["cluster_metrics"]
        s_acc_str = f"{sm['accuracy']*100:.2f}% [{sm['accuracy_ci_95'][0]*100:.1f}-{sm['accuracy_ci_95'][1]*100:.1f}]"
        s_st_str = f"{sm['stone_recall']*100:.2f}% [{sm['stone_recall_ci_95'][0]*100:.1f}-{sm['stone_recall_ci_95'][1]*100:.1f}]"
        s_tu_str = f"{sm['tumor_recall']*100:.2f}% [{sm['tumor_recall_ci_95'][0]*100:.1f}-{sm['tumor_recall_ci_95'][1]*100:.1f}]"
        c_acc_str = f"{cm['accuracy']*100:.2f}% [{cm['accuracy_ci_95'][0]*100:.1f}-{cm['accuracy_ci_95'][1]*100:.1f}]"
        c_st_str = f"{cm['stone_recall']*100:.2f}% [{cm['stone_recall_ci_95'][0]*100:.1f}-{cm['stone_recall_ci_95'][1]*100:.1f}]"
        c_tu_str = f"{cm['tumor_recall']*100:.2f}% [{cm['tumor_recall_ci_95'][0]*100:.1f}-{cm['tumor_recall_ci_95'][1]*100:.1f}]"
        print(f"{var:<22} | {s_acc_str:<26} | {s_st_str:<22} | {s_tu_str:<22} | {c_acc_str:<26} | {c_st_str:<14} | {c_tu_str:<14} | {data['total_time_sec']:<8}", flush=True)
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
