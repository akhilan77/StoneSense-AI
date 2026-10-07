"""StoneSense-AI Checkpoint Evaluation Service.

Evaluates a ResNet18 model checkpoint against the central validation, test, or pooled dataset.
Computes slice-level and cluster-level (mean-probability vote) metrics, per-class recall,
and cluster-bootstrap 95% Confidence Intervals.

Supports dry-run inspection (default) and ORM-managed database updates (--write-db).
Guarantees: Never modifies deployment status or sets is_deployed=True.
"""

import sys
import os
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "backend"))

from model import build_resnet18_classifier, CLASS_MAPPING
from transforms import get_val_test_transforms
from app.db.database import SessionLocal
from app.db.models import ModelVersion
from app.services.deployment_gate import compute_validation_dataset_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalCheckpoint")

MANIFEST_PATH = PROJECT_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"


class SliceImageDataset(Dataset):
    """Dataset that returns image tensor, label index, and image filename for cluster mapping."""

    def __init__(self, file_tuples: List[Tuple[Path, int]], transform=None):
        self.file_tuples = file_tuples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_tuples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, target = self.file_tuples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path.name


def build_split_file_list(split: str = "validation") -> List[Tuple[Path, int]]:
    """Builds list of (file_path, class_idx) for the requested split(s)."""
    splits_to_load = ["validation", "test"] if split == "pooled" else [split]
    partitions_dir = PROJECT_ROOT / "dl" / "datasets" / "partitions"
    hospitals = ["hospital_1", "hospital_2", "hospital_3"]
    class_to_idx = {name: idx for idx, name in CLASS_MAPPING.items()}

    file_tuples = []
    for s in splits_to_load:
        for h in hospitals:
            for cls_idx, cls_name in CLASS_MAPPING.items():
                folder = partitions_dir / h / s / cls_name
                if folder.exists():
                    for img_p in folder.glob("*.*"):
                        if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            file_tuples.append((img_p, cls_idx))

    if not file_tuples:
        # Fallback to dl/processed_grouped
        for s in splits_to_load:
            for cls_idx, cls_name in CLASS_MAPPING.items():
                folder = PROJECT_ROOT / "dl" / "processed_grouped" / s / cls_name
                if folder.exists():
                    for img_p in folder.glob("*.*"):
                        if img_p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            file_tuples.append((img_p, cls_idx))

    if not file_tuples:
        raise FileNotFoundError(f"No image files found for split: '{split}'")

    return file_tuples



def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str = "validation",
    device_str: Optional[str] = None,
    batch_size: int = 64,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """Evaluates checkpoint file across slices and clusters, calculating bootstrap CIs."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if split == "pooled":
        logger.warning("=" * 80)
        logger.warning("[WARNING] NOTICE: Evaluating on pooled (validation + test) partitions.")
        logger.warning("[WARNING] The test set is no longer untouched and cannot serve as an independent held-out benchmark.")
        logger.warning("=" * 80)


    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Loading checkpoint '{checkpoint_path.name}' for evaluation on '{split}' split ({device})...")

    # Load cluster manifest
    manifest_df = pd.read_csv(MANIFEST_PATH)
    file_to_cluster = dict(zip(manifest_df["filename"], manifest_df["near_duplicate_group_id"]))

    # Build model architecture
    model = build_resnet18_classifier(num_classes=4, freeze_backbone=False)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load dataset
    val_tf = get_val_test_transforms(image_size=(224, 224))
    file_tuples = build_split_file_list(split=split)
    dataset = SliceImageDataset(file_tuples, transform=val_tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    records = []
    with torch.no_grad():
        for imgs, targets, filenames in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            targets = targets.numpy()

            for fn, y_true, y_pred, p_vec in zip(filenames, targets, preds, probs):
                cid = file_to_cluster.get(fn, f"SINGLETON_{fn}")
                records.append({
                    "filename": fn,
                    "cluster_id": cid,
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "prob_cyst": float(p_vec[0]),
                    "prob_normal": float(p_vec[1]),
                    "prob_stone": float(p_vec[2]),
                    "prob_tumor": float(p_vec[3]),
                })

    df = pd.DataFrame(records)

    # 1. Slice-Level Metrics
    y_true_slices = df["y_true"].values
    y_pred_slices = df["y_pred"].values
    slice_acc = float(accuracy_score(y_true_slices, y_pred_slices))
    slice_f1 = float(f1_score(y_true_slices, y_pred_slices, average="macro"))
    slice_prec = float(precision_score(y_true_slices, y_pred_slices, average="macro", zero_division=0))
    slice_rec = float(recall_score(y_true_slices, y_pred_slices, average="macro", zero_division=0))
    slice_per_class_rec = recall_score(y_true_slices, y_pred_slices, average=None, labels=[0, 1, 2, 3], zero_division=0)
    slice_per_class_prec = precision_score(y_true_slices, y_pred_slices, average=None, labels=[0, 1, 2, 3], zero_division=0)
    slice_cm = confusion_matrix(y_true_slices, y_pred_slices, labels=[0, 1, 2, 3]).tolist()

    # 2. Cluster-Level (Mean-Probability Vote) Metrics
    cluster_records = []
    prob_cols = ["prob_cyst", "prob_normal", "prob_stone", "prob_tumor"]
    for cid, cgroup in df.groupby("cluster_id"):
        mean_probs = cgroup[prob_cols].mean().values
        c_pred = int(np.argmax(mean_probs))
        # Ground truth is majority label within the cluster
        c_true = int(cgroup["y_true"].mode().iloc[0])
        cluster_records.append({
            "cluster_id": cid,
            "y_true": c_true,
            "y_pred": c_pred,
            "slices": len(cgroup)
        })

    cdf = pd.DataFrame(cluster_records)
    y_true_clusters = cdf["y_true"].values
    y_pred_clusters = cdf["y_pred"].values

    cluster_acc = float(accuracy_score(y_true_clusters, y_pred_clusters))
    cluster_f1 = float(f1_score(y_true_clusters, y_pred_clusters, average="macro"))
    cluster_per_class_rec = recall_score(y_true_clusters, y_pred_clusters, average=None, labels=[0, 1, 2, 3], zero_division=0)
    cluster_cm = confusion_matrix(y_true_clusters, y_pred_clusters, labels=[0, 1, 2, 3]).tolist()

    # 3. Cluster-Level Non-Parametric Bootstrap 95% Confidence Intervals
    np.random.seed(42)
    unique_clusters = df["cluster_id"].unique()
    n_clusters = len(unique_clusters)

    boot_slice_accs, boot_slice_stone_recs, boot_slice_tumor_recs = [], [], []
    boot_cluster_accs, boot_cluster_stone_recs, boot_cluster_tumor_recs = [], [], []

    # Pre-aggregate cluster dfs for high-speed vector bootstrap
    cluster_df_dict = {cid: grp for cid, grp in df.groupby("cluster_id")}
    cluster_meta_dict = {row["cluster_id"]: (row["y_true"], row["y_pred"]) for _, row in cdf.iterrows()}

    for _ in range(n_bootstrap):
        sampled_cids = np.random.choice(unique_clusters, size=n_clusters, replace=True)
        
        # Slices from sampled clusters
        sampled_slice_dfs = [cluster_df_dict[cid] for cid in sampled_cids]
        boot_slice_df = pd.concat(sampled_slice_dfs, ignore_index=True)
        b_true_s = boot_slice_df["y_true"].values
        b_pred_s = boot_slice_df["y_pred"].values
        boot_slice_accs.append(accuracy_score(b_true_s, b_pred_s))
        s_recs = recall_score(b_true_s, b_pred_s, average=None, labels=[2, 3], zero_division=0)
        boot_slice_stone_recs.append(s_recs[0])
        boot_slice_tumor_recs.append(s_recs[1])

        # Sampled cluster predictions
        b_true_c = [cluster_meta_dict[cid][0] for cid in sampled_cids]
        b_pred_c = [cluster_meta_dict[cid][1] for cid in sampled_cids]
        boot_cluster_accs.append(accuracy_score(b_true_c, b_pred_c))
        c_recs = recall_score(b_true_c, b_pred_c, average=None, labels=[2, 3], zero_division=0)
        boot_cluster_stone_recs.append(c_recs[0])
        boot_cluster_tumor_recs.append(c_recs[1])

    ci_slice_acc = [float(np.percentile(boot_slice_accs, 2.5)), float(np.percentile(boot_slice_accs, 97.5))]
    ci_slice_stone = [float(np.percentile(boot_slice_stone_recs, 2.5)), float(np.percentile(boot_slice_stone_recs, 97.5))]
    ci_slice_tumor = [float(np.percentile(boot_slice_tumor_recs, 2.5)), float(np.percentile(boot_slice_tumor_recs, 97.5))]

    ci_cluster_acc = [float(np.percentile(boot_cluster_accs, 2.5)), float(np.percentile(boot_cluster_accs, 97.5))]
    ci_cluster_stone = [float(np.percentile(boot_cluster_stone_recs, 2.5)), float(np.percentile(boot_cluster_stone_recs, 97.5))]
    ci_cluster_tumor = [float(np.percentile(boot_cluster_tumor_recs, 2.5)), float(np.percentile(boot_cluster_tumor_recs, 97.5))]

    runtime_hash = compute_validation_dataset_hash()

    results = {
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "total_slices": len(df),
        "total_clusters": n_clusters,
        "data_source_hash": runtime_hash,
        "is_pooled_test_touched": split == "pooled",
        
        # Slice-level
        "slice_accuracy": round(slice_acc, 6),
        "slice_accuracy_ci": [round(x, 4) for x in ci_slice_acc],
        "slice_f1_macro": round(slice_f1, 6),
        "slice_precision_macro": round(slice_prec, 6),
        "slice_recall_macro": round(slice_rec, 6),
        "slice_recall_stone": round(float(slice_per_class_rec[2]), 6),
        "slice_recall_stone_ci": [round(x, 4) for x in ci_slice_stone],
        "slice_recall_tumor": round(float(slice_per_class_rec[3]), 6),
        "slice_recall_tumor_ci": [round(x, 4) for x in ci_slice_tumor],
        "slice_per_class_recall": {CLASS_MAPPING[i]: round(float(slice_per_class_rec[i]), 4) for i in range(4)},
        "slice_per_class_precision": {CLASS_MAPPING[i]: round(float(slice_per_class_prec[i]), 4) for i in range(4)},
        "slice_confusion_matrix": slice_cm,

        # Cluster-level
        "cluster_accuracy": round(cluster_acc, 6),
        "cluster_accuracy_ci": [round(x, 4) for x in ci_cluster_acc],
        "cluster_f1_macro": round(cluster_f1, 6),
        "cluster_recall_stone": round(float(cluster_per_class_rec[2]), 6),
        "cluster_recall_stone_ci": [round(x, 4) for x in ci_cluster_stone],
        "cluster_recall_tumor": round(float(cluster_per_class_rec[3]), 6),
        "cluster_recall_tumor_ci": [round(x, 4) for x in ci_cluster_tumor],
        "cluster_per_class_recall": {CLASS_MAPPING[i]: round(float(cluster_per_class_rec[i]), 4) for i in range(4)},
        "cluster_confusion_matrix": cluster_cm,
    }
    return results


def print_evaluation_report(results: Dict[str, Any]):
    """Prints a formatted evaluation table with slice and cluster metrics and bootstrap CIs."""
    print("\n" + "=" * 80)
    print(f"CHECKPOINT EVALUATION REPORT: {results['checkpoint_name']}")
    print(f"Split: {results['split'].upper()} | Slices: {results['total_slices']} | Clusters: {results['total_clusters']} | Hash: {results['data_source_hash']}")
    if results.get("is_pooled_test_touched"):
        print("[WARNING] NOTICE: Pooled (validation + test) evaluation. The test set is no longer untouched.")
    print("=" * 80)

    
    print("\n1. SLICE-LEVEL EVALUATION (with 95% Cluster-Bootstrap CIs):")
    print("-" * 80)
    print(f"  Slice Accuracy:        {results['slice_accuracy']:.4f} ({results['slice_accuracy']*100:.2f}%)  [95% CI: {results['slice_accuracy_ci'][0]:.4f} - {results['slice_accuracy_ci'][1]:.4f}]")
    print(f"  Macro F1 Score:        {results['slice_f1_macro']:.4f}")
    print(f"  Macro Precision:       {results['slice_precision_macro']:.4f}")
    print(f"  Macro Recall:          {results['slice_recall_macro']:.4f}")
    print(f"  Stone Recall (Sens):   {results['slice_recall_stone']:.4f}  [95% CI: {results['slice_recall_stone_ci'][0]:.4f} - {results['slice_recall_stone_ci'][1]:.4f}]")
    print(f"  Tumor Recall (Sens):   {results['slice_recall_tumor']:.4f}  [95% CI: {results['slice_recall_tumor_ci'][0]:.4f} - {results['slice_recall_tumor_ci'][1]:.4f}]")
    print("\n  Per-Class Slice Recalls:")
    for cls_name, rec in results["slice_per_class_recall"].items():
        prec = results["slice_per_class_precision"][cls_name]
        print(f"    {cls_name:<8} Recall: {rec:.4f} | Precision: {prec:.4f}")

    print("\n2. CLUSTER-LEVEL VOTING EVALUATION (Mean-Probability Vote per Patient/Cluster):")
    print("-" * 80)
    print(f"  Cluster Accuracy:      {results['cluster_accuracy']:.4f} ({results['cluster_accuracy']*100:.2f}%)  [95% CI: {results['cluster_accuracy_ci'][0]:.4f} - {results['cluster_accuracy_ci'][1]:.4f}]")
    print(f"  Cluster Macro F1:      {results['cluster_f1_macro']:.4f}")
    print(f"  Stone Recall (Sens):   {results['cluster_recall_stone']:.4f}  [95% CI: {results['cluster_recall_stone_ci'][0]:.4f} - {results['cluster_recall_stone_ci'][1]:.4f}]")
    print(f"  Tumor Recall (Sens):   {results['cluster_recall_tumor']:.4f}  [95% CI: {results['cluster_recall_tumor_ci'][0]:.4f} - {results['cluster_recall_tumor_ci'][1]:.4f}]")
    print("\n  Per-Class Cluster Recalls:")
    for cls_name, rec in results["cluster_per_class_recall"].items():
        print(f"    {cls_name:<8} Recall: {rec:.4f}")

    print("\n" + "=" * 80)


def write_to_model_versions_db(
    results: Dict[str, Any],
    version_tag: Optional[str] = None,
    model_family: str = "resnet18_ct",
    round_id: Optional[int] = None
):
    """Persists central evaluation metrics to the model_versions table via SQLAlchemy ORM."""
    tag = version_tag or Path(results["checkpoint_path"]).stem
    db = SessionLocal()
    try:
        mv = db.query(ModelVersion).filter_by(version_tag=tag, model_family=model_family).first()
        now_iso = datetime.now(timezone.utc).isoformat()

        ckpt_path = Path(results["checkpoint_path"])
        ckpt_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else None

        gate_payload = {
            "is_central_eval": True,
            "checkpoint_sha256": ckpt_sha256,
            "validation_set_hash": results["data_source_hash"],
            "data_source": results["data_source_hash"],
            "split": results["split"],
            "accuracy": results["slice_accuracy"],
            "accuracy_ci": results["slice_accuracy_ci"],
            "f1_macro": results["slice_f1_macro"],
            "precision_macro": results["slice_precision_macro"],
            "recall_macro": results["slice_recall_macro"],
            "recall_stone": results["slice_recall_stone"],
            "recall_stone_ci": results["slice_recall_stone_ci"],
            "recall_tumor": results["slice_recall_tumor"],
            "recall_tumor_ci": results["slice_recall_tumor_ci"],
            "cluster_accuracy": results["cluster_accuracy"],
            "cluster_accuracy_ci": results["cluster_accuracy_ci"],
            "cluster_f1_macro": results["cluster_f1_macro"],
            "slice_per_class_recall": results["slice_per_class_recall"],
            "cluster_per_class_recall": results["cluster_per_class_recall"],
            "confusion_matrix": results["slice_confusion_matrix"],
            "evaluated_at": now_iso,
        }

        if mv is not None:
            logger.info(f"Updating existing ModelVersion row (ID: {mv.id}, tag: '{mv.version_tag}') with central-eval metrics...")
            mv.accuracy = results["slice_accuracy"]
            mv.f1_score = results["slice_f1_macro"]
            mv.precision = results["slice_precision_macro"]
            mv.recall = results["slice_recall_macro"]
            mv.gate_report = gate_payload
        else:
            logger.info(f"Creating new ModelVersion row for tag '{tag}' (status: pending_review, is_deployed: False)...")
            mv = ModelVersion(
                model_family=model_family,
                version_tag=tag,
                round_id=round_id,
                accuracy=results["slice_accuracy"],
                f1_score=results["slice_f1_macro"],
                precision=results["slice_precision_macro"],
                recall=results["slice_recall_macro"],
                is_deployed=False,
                status="pending_review",
                artifact_path=str(results["checkpoint_path"]),
                gate_report=gate_payload,
                trained_at=datetime.now(timezone.utc),
            )
            db.add(mv)

        db.commit()
        db.refresh(mv)
        logger.info(f"Successfully stored central-eval metrics for '{tag}' (ModelVersion ID: {mv.id}, status: '{mv.status}', is_deployed: {mv.is_deployed})")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a ResNet18 checkpoint on central dataset split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test", "pooled"], help="Evaluation split")
    parser.add_argument("--version-tag", type=str, default=None, help="Version tag for ModelVersion registry")
    parser.add_argument("--model-family", type=str, default="resnet18_ct", help="Model family identifier")
    parser.add_argument("--round-id", type=int, default=None, help="FL Round ID if applicable")
    parser.add_argument("--write-db", action="store_true", help="Persist central evaluation metrics to model_versions table")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device (cpu/cuda)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap iterations")

    args = parser.parse_args()
    eval_res = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        device_str=args.device,
        n_bootstrap=args.n_boot
    )
    print_evaluation_report(eval_res)

    if args.write_db:
        write_to_model_versions_db(
            results=eval_res,
            version_tag=args.version_tag,
            model_family=args.model_family,
            round_id=args.round_id,
        )
    else:
        print("[DRY-RUN] Database write skipped. Pass --write-db to persist central metrics to model_versions.")
