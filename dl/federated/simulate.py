"""StoneSense-AI Federated Learning Simulation Runner.

Executes multi-hospital Federated Learning simulation across N rounds.
Orchestrates isolated client training on hospital partitions, FedAvg aggregation,
model checkpointing, and database telemetry persistence.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import time
import logging
import argparse
from typing import Dict, List, Any
import json
import numpy as np
import torch
torch.set_num_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

from model import build_resnet18_classifier, CLASS_MAPPING
from partition import partition_dataset, HOSPITAL_IDS, HOSPITAL_NAMES
from client import StoneSenseFLClient
from strategy import persist_round_to_db
from local_training import get_model_parameters, set_model_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("StoneSenseFLSimulation")


def aggregate_weights(
    client_weights: List[List[np.ndarray]],
    sample_counts: List[int]
) -> List[np.ndarray]:
    """Federated Averaging (FedAvg) parameter aggregation."""
    total_samples = sum(sample_counts)
    if total_samples == 0:
        return client_weights[0]

    num_layers = len(client_weights[0])
    aggregated = []

    for layer_idx in range(num_layers):
        layer_weighted_sum = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for client_idx, weights in enumerate(client_weights):
            weight = sample_counts[client_idx] / total_samples
            layer_weighted_sum += weight * weights[layer_idx].astype(np.float64)
        aggregated.append(layer_weighted_sum.astype(client_weights[0][layer_idx].dtype))

    return aggregated


def sync_hospital_dataset_metadata(partitions_meta: Dict[str, Any]) -> None:
    """Synchronizes partition sizes and class distributions to the DB."""
    try:
        from app.db.database import SessionLocal, init_db
        from app.db.models import Hospital
    except ImportError as e:
        logger.warning(f"Could not import backend DB: {e}")
        return

    init_db()
    db = SessionLocal()
    code_map = {"hospital_1": "HOSP-001", "hospital_2": "HOSP-002", "hospital_3": "HOSP-003"}
    try:
        dist = partitions_meta.get("distributions", {})
        for h_id, splits in dist.items():
            db_code = code_map.get(h_id, h_id)
            name = partitions_meta.get("hospital_names", {}).get(h_id, f"Hospital {db_code}")
            train_dist = splits.get("train", {})
            total_size = sum(train_dist.values()) + sum(splits.get("validation", {}).values())

            h = db.query(Hospital).filter_by(hospital_code=db_code).first()
            if not h:
                h = Hospital(
                    hospital_code=db_code,
                    name=name,
                    region="Simulation Node",
                    dataset_size=total_size,
                    class_distribution=train_dist,
                    is_active=True
                )
                db.add(h)
            else:
                h.dataset_size = total_size
                h.class_distribution = train_dist
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed syncing hospital metadata: {e}")
    finally:
        db.close()


def run_federated_simulation(
    rounds: int = 3,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.0005,
    mode: str = "iid",
    seed: int = 42,
    device: str = "auto"
) -> Dict[str, Any]:
    """Runs the full multi-hospital federated training simulation."""
    logger.info("=" * 70)
    logger.info("Starting StoneSense-AI Multi-Hospital Federated Learning Simulation")
    logger.info(f"Rounds: {rounds} | Local Epochs: {local_epochs} | Mode: {mode.upper()} | Seed: {seed}")
    logger.info("=" * 70)

    # 1. Verify / generate partitions
    partitions_root = PROJECT_ROOT / "dl" / "datasets" / "partitions"
    dist_file = partitions_root / "distribution.json"
    if not dist_file.exists():
        logger.info("Partitions not found. Generating hospital partitions...")
        partitions_meta = partition_dataset(
            source_dir=PROJECT_ROOT / "dl" / "processed",
            output_dir=partitions_root,
            mode=mode,
            seed=seed
        )
    else:
        with open(dist_file, "r", encoding="utf-8") as f:
            partitions_meta = json.load(f)

    sync_hospital_dataset_metadata(partitions_meta)

    # 2. Setup Device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    logger.info(f"Running simulation on device: {dev}")

    # 3. Instantiate Hospital Clients
    clients: List[StoneSenseFLClient] = []
    for h_id in HOSPITAL_IDS:
        client = StoneSenseFLClient(
            hospital_id=h_id,
            partition_dir=partitions_root / h_id,
            batch_size=batch_size,
            device=dev,
            lr=lr
        )
        clients.append(client)

    # 4. Initialize Global Model
    class_names = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
    global_model = build_resnet18_classifier(
        num_classes=len(class_names),
        freeze_backbone=True,
        unfreeze_layer4=True
    ).to(dev)

    global_weights = get_model_parameters(global_model)

    history = {
        "rounds": [],
        "global_accuracy": [],
        "global_f1": [],
        "global_loss": []
    }

    # 5. Execute Federated Rounds
    for round_num in range(1, rounds + 1):
        round_start = time.time()
        logger.info(f"\n>>> Starting Federated Round {round_num}/{rounds} <<<")

        client_weights = []
        client_sample_counts = []
        client_fit_metrics = []
        client_eval_metrics = []

        # Local training on each hospital node
        config = {"current_round": round_num, "local_epochs": local_epochs, "lr": lr}

        for client in clients:
            updated_weights, samples, metrics = client.fit(global_weights, config)
            client_weights.append(updated_weights)
            client_sample_counts.append(samples)
            client_fit_metrics.append(metrics)

        # FedAvg Aggregation
        logger.info(f"[Round {round_num}] Aggregating weights across {len(clients)} hospital clients...")
        global_weights = aggregate_weights(client_weights, client_sample_counts)
        set_model_parameters(global_model, global_weights)

        # Global Model Evaluation across each hospital validation split
        total_eval_samples = 0
        weighted_eval_loss = 0.0
        weighted_eval_acc = 0.0
        weighted_eval_f1 = 0.0
        weighted_eval_prec = 0.0
        weighted_eval_rec = 0.0

        for client in clients:
            loss, samples, eval_metrics = client.evaluate(global_weights, config)
            client_eval_metrics.append(eval_metrics)

            total_eval_samples += samples
            weighted_eval_loss += loss * samples
            weighted_eval_acc += float(eval_metrics["accuracy"]) * samples
            weighted_eval_f1 += float(eval_metrics["f1_macro"]) * samples
            weighted_eval_prec += float(eval_metrics["precision_macro"]) * samples
            weighted_eval_rec += float(eval_metrics["recall_macro"]) * samples

        global_val_loss = weighted_eval_loss / max(total_eval_samples, 1)
        global_val_acc = weighted_eval_acc / max(total_eval_samples, 1)
        global_val_f1 = weighted_eval_f1 / max(total_eval_samples, 1)
        global_val_prec = weighted_eval_prec / max(total_eval_samples, 1)
        global_val_rec = weighted_eval_rec / max(total_eval_samples, 1)

        # Aggregate Fit Metrics
        tot_train_samples = sum(client_sample_counts)
        global_train_loss = sum(m["train_loss"] * client_sample_counts[i] for i, m in enumerate(client_fit_metrics)) / max(tot_train_samples, 1)
        global_train_acc = sum(m["train_accuracy"] * client_sample_counts[i] for i, m in enumerate(client_fit_metrics)) / max(tot_train_samples, 1)

        fit_summary = {"train_loss": global_train_loss, "train_accuracy": global_train_acc}
        eval_summary = {
            "loss": global_val_loss,
            "accuracy": global_val_acc,
            "f1_macro": global_val_f1,
            "precision_macro": global_val_prec,
            "recall_macro": global_val_rec
        }

        # Build client runs telemetry list
        client_runs = []
        for i, client in enumerate(clients):
            fit_m = client_fit_metrics[i]
            eval_m = client_eval_metrics[i]
            client_runs.append({
                "hospital_code": client.hospital_id,
                "train_loss": fit_m.get("train_loss"),
                "train_acc": fit_m.get("train_accuracy"),
                "train_f1": fit_m.get("train_f1_macro"),
                "val_loss": eval_m.get("loss"),
                "val_acc": eval_m.get("accuracy"),
                "val_f1": eval_m.get("f1_macro"),
                "sample_count": client_sample_counts[i],
            })

        duration = time.time() - round_start

        # Persist to Database & Save Checkpoint
        persist_round_to_db(
            round_number=round_num,
            mode=mode,
            parameters=global_weights,
            fit_metrics=fit_summary,
            eval_metrics=eval_summary,
            client_runs=client_runs,
            duration_sec=duration
        )

        history["rounds"].append(round_num)
        history["global_accuracy"].append(round(global_val_acc, 4))
        history["global_f1"].append(round(global_val_f1, 4))
        history["global_loss"].append(round(global_val_loss, 4))

        logger.info(
            f"=== Round {round_num} Result: Global Val Acc: {global_val_acc:.4f} | "
            f"Global Val F1: {global_val_f1:.4f} | Loss: {global_val_loss:.4f} | Duration: {duration:.2f}s ==="
        )

    logger.info("\n" + "=" * 70)
    logger.info("Federated Learning Simulation Completed Successfully!")
    logger.info(f"Rounds Completed: {rounds}")
    logger.info(f"Final Global Accuracy: {history['global_accuracy'][-1]:.4f}")
    logger.info(f"Final Global Macro F1: {history['global_f1'][-1]:.4f}")
    logger.info("=" * 70)

    return history


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Run StoneSense-AI FL Simulation.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--mode", type=str, default="iid", choices=["iid", "non-iid"], help="Distribution mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, or auto)")

    args = parser.parse_args()

    run_federated_simulation(
        rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode,
        seed=args.seed,
        device=args.device
    )
