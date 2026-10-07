"""StoneSense-AI Checkpoint Evaluation Service.

Evaluates a ResNet18 model checkpoint against the central validation or test dataset.
Computes global accuracy, macro F1, per-class sensitivity (recall), precision, support,
confusion matrix, and dataset runtime signature hash.

Supports dry-run inspection (default) and ORM-managed database updates (--write-db).
Guarantees: Never modifies deployment status or sets is_deployed=True.
"""

import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "backend"))

from model import build_resnet18_classifier, CLASS_MAPPING
from transforms import get_val_test_transforms
from evaluate_helpers import evaluate_model
from app.db.database import SessionLocal
from app.db.models import ModelVersion
from app.services.deployment_gate import compute_validation_dataset_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalCheckpoint")


def build_central_split_dataset(split: str = "validation"):
    """Builds a concatenated PyTorch dataset across all hospital partitions for the given split."""
    val_tf = get_val_test_transforms(image_size=(224, 224))
    partitions_dir = PROJECT_ROOT / "dl" / "datasets" / "partitions"
    hospitals = ["hospital_1", "hospital_2", "hospital_3"]

    datasets_list = []
    for h in hospitals:
        split_dir = partitions_dir / h / split
        if split_dir.exists():
            datasets_list.append(datasets.ImageFolder(root=str(split_dir), transform=val_tf))

    if not datasets_list:
        fallback_dir = PROJECT_ROOT / "dl" / "processed_grouped" / split
        if fallback_dir.exists():
            datasets_list.append(datasets.ImageFolder(root=str(fallback_dir), transform=val_tf))

    if not datasets_list:
        raise FileNotFoundError(f"No dataset partitions found for split '{split}'")

    return ConcatDataset(datasets_list) if len(datasets_list) > 1 else datasets_list[0]


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str = "validation",
    device_str: Optional[str] = None,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Evaluates checkpoint file and returns comprehensive metrics dict."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Loading checkpoint '{checkpoint_path.name}' for evaluation on '{split}' split ({device})...")

    # Build model architecture
    model = build_resnet18_classifier(num_classes=4, freeze_backbone=False)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any("layer" in k or "fc" in k for k in ckpt.keys()):
        state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create central dataset and loader
    dataset = build_central_split_dataset(split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate
    class_names = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
    metrics, _, _, _ = evaluate_model(model, dataloader, device, class_names)
    runtime_hash = compute_validation_dataset_hash()

    results = {
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "split": split,
        "total_samples": len(dataset),
        "data_source_hash": runtime_hash,
        "accuracy": round(float(metrics["accuracy"]), 6),
        "f1_macro": round(float(metrics["f1_macro"]), 6),
        "precision_macro": round(float(metrics["precision_macro"]), 6),
        "recall_macro": round(float(metrics["recall_macro"]), 6),
        "recall_stone": round(float(metrics["per_class_metrics"]["Stone"]["recall"]), 6),
        "recall_tumor": round(float(metrics["per_class_metrics"]["Tumor"]["recall"]), 6),
        "precision_stone": round(float(metrics["per_class_metrics"]["Stone"]["precision"]), 6),
        "per_class_metrics": metrics["per_class_metrics"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    return results


def print_evaluation_report(results: Dict[str, Any]):
    """Prints a formatted evaluation table and confusion matrix."""
    print("=" * 70)
    print(f"CHECKPOINT EVALUATION REPORT: {results['checkpoint_name']}")
    print(f"Split: {results['split'].upper()} | Samples: {results['total_samples']} | Hash: {results['data_source_hash']}")
    print("-" * 70)
    print(f"  Overall Accuracy:    {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Macro F1 Score:      {results['f1_macro']:.4f}")
    print(f"  Macro Precision:     {results['precision_macro']:.4f}")
    print(f"  Macro Recall:        {results['recall_macro']:.4f}")
    print(f"  Stone Recall (Sens): {results['recall_stone']:.4f}")
    print(f"  Tumor Recall (Sens): {results['recall_tumor']:.4f}")
    print(f"  Stone Precision:     {results['precision_stone']:.4f}")
    print("-" * 70)
    print("PER-CLASS BREAKDOWN:")
    print(f"  {'Class':<10} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Support':<8}")
    for cls_name, cm in results["per_class_metrics"].items():
        print(f"  {cls_name:<10} {cm['recall']:<10.4f} {cm['precision']:<12.4f} {cm['f1_score']:<10.4f} {cm['support']:<8}")
    print("-" * 70)
    print("CONFUSION MATRIX (Row=True, Col=Pred) [Cyst, Normal, Stone, Tumor]:")
    for row_idx, row in enumerate(results["confusion_matrix"]):
        cls_name = list(results["per_class_metrics"].keys())[row_idx]
        print(f"  {cls_name:<8} {row}")
    print("=" * 70)


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
        now_iso = datetime.utcnow().isoformat()

        ckpt_path = Path(results["checkpoint_path"])
        ckpt_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path.exists() else None

        gate_payload = {
            "is_central_eval": True,
            "checkpoint_sha256": ckpt_sha256,
            "validation_set_hash": results["data_source_hash"],
            "data_source": results["data_source_hash"],
            "accuracy": results["accuracy"],
            "f1_macro": results["f1_macro"],
            "precision_macro": results["precision_macro"],
            "recall_macro": results["recall_macro"],
            "recall_stone": results["recall_stone"],
            "recall_tumor": results["recall_tumor"],
            "precision_stone": results["precision_stone"],
            "per_class_metrics": results["per_class_metrics"],
            "confusion_matrix": results["confusion_matrix"],
            "evaluated_at": now_iso,
        }

        if mv is not None:
            logger.info(f"Updating existing ModelVersion row (ID: {mv.id}, tag: '{mv.version_tag}') with central-eval metrics...")
            mv.accuracy = results["accuracy"]
            mv.f1_score = results["f1_macro"]
            mv.precision = results["precision_macro"]
            mv.recall = results["recall_macro"]
            mv.gate_report = gate_payload
            # Preserves deployment status - NEVER sets deployed
        else:
            logger.info(f"Creating new ModelVersion row for tag '{tag}' (status: pending_review, is_deployed: False)...")
            mv = ModelVersion(
                model_family=model_family,
                version_tag=tag,
                round_id=round_id,
                accuracy=results["accuracy"],
                f1_score=results["f1_macro"],
                precision=results["precision_macro"],
                recall=results["recall_macro"],
                is_deployed=False,
                status="pending_review",
                artifact_path=str(results["checkpoint_path"]),
                gate_report=gate_payload,
                trained_at=datetime.utcnow(),
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
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"], help="Evaluation split")
    parser.add_argument("--version-tag", type=str, default=None, help="Version tag for ModelVersion registry")
    parser.add_argument("--model-family", type=str, default="resnet18_ct", help="Model family identifier")
    parser.add_argument("--round-id", type=int, default=None, help="FL Round ID if applicable")
    parser.add_argument("--write-db", action="store_true", help="Persist central evaluation metrics to model_versions table")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device (cpu/cuda)")

    args = parser.parse_args()
    eval_res = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        device_str=args.device,
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
