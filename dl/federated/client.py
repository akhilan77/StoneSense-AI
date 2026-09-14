"""StoneSense-AI Hospital Federated Client (Flower NumPyClient).

Runs on individual hospital infrastructure in a decentralized multi-tenant deployment.
Trains ResNet18 locally on hospital CT partition and transmits ONLY model weights.
Guarantees zero raw patient CT scan transmission outside the hospital boundary.
"""

from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

try:
    import flwr as fl
    _NumPyClientBase = fl.client.NumPyClient
except ImportError:
    fl = None
    class _NumPyClientBase:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

from model import build_resnet18_classifier, CLASS_MAPPING
from local_training import (
    get_model_parameters, set_model_parameters,
    train_local, evaluate_local
)
from dataloaders import create_dataloaders
from transforms import get_train_transforms, get_val_test_transforms

logger = logging.getLogger("StoneSenseFLClient")


class StoneSenseFLClient(_NumPyClientBase):
    """Flower NumPyClient for a specific hospital partition."""

    def __init__(
        self,
        hospital_id: str,
        partition_dir: Path,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        lr: float = 0.001
    ):
        self.hospital_id = hospital_id
        self.partition_dir = Path(partition_dir)
        self.batch_size = batch_size
        self.lr = lr

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.class_names = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]

        # Initialize local model
        self.model = build_resnet18_classifier(
            num_classes=len(self.class_names),
            freeze_backbone=True,
            unfreeze_layer4=True
        ).to(self.device)

        # Load local DataLoaders
        train_tf = get_train_transforms(image_size=(224, 224))
        val_tf = get_val_test_transforms(image_size=(224, 224))

        dataloaders = create_dataloaders(
            processed_dir=self.partition_dir,
            train_transform=train_tf,
            val_test_transform=val_tf,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["validation"]
        self.test_loader = dataloaders["test"]

        logger.info(
            f"[{self.hospital_id}] Initialized with {len(self.train_loader.dataset)} train samples, "
            f"{len(self.val_loader.dataset)} val samples on {self.device}"
        )

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Returns the local model parameters as NumPy ndarrays."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets the local model parameters from NumPy ndarrays."""
        set_model_parameters(self.model, parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Trains model on local hospital partition."""
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", self.lr))
        current_round = int(config.get("current_round", 1))
        h_label = {"hospital_1": "Hospital 1", "hospital_2": "Hospital 2", "hospital_3": "Hospital 3"}.get(self.hospital_id, self.hospital_id)
        prev_ver = f"v{current_round-1}" if current_round > 1 else "v1 (initial)"

        print("")
        print("-" * 55)
        print(f"[{h_label}] FEDERATED ROUND {current_round}")
        print(f"[{h_label}] Received global model: {prev_ver}")
        print(f"[{h_label}] Privacy guarantee: RAW CT DATA REMAINS LOCAL (never transmitted)")
        print(f"[{h_label}] Local ResNet18 training started on {len(self.train_loader.dataset)} CT scans ({local_epochs} epoch(s))...")

        self.model, metrics, sample_count = train_local(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=local_epochs,
            lr=lr,
            device=self.device,
            class_names=self.class_names
        )

        print(f"[{h_label}] Local training completed:")
        print(f"  - Local Loss:     {metrics['train_loss']:.4f}")
        print(f"  - Local Accuracy: {metrics['train_accuracy']*100:.2f}%")
        print(f"  - Local F1:       {metrics['train_f1_macro']*100:.2f}%")
        if 'val_accuracy' in metrics:
            print(f"  - Local Val Acc:  {metrics['val_accuracy']*100:.2f}%")
            print(f"  - Local Val F1:   {metrics['val_f1_macro']*100:.2f}%")
        print(f"[{h_label}] Model weights & evaluation metrics being returned to Flower server")
        print("-" * 55)
        print("")

        # Include hospital_id in metrics for telemetry and logging
        metrics["hospital_id"] = self.hospital_id
        metrics["current_round"] = current_round

        # Ensure all values are scalar primitives
        clean_metrics = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in metrics.items()}

        return get_model_parameters(self.model), sample_count, clean_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluates global model on local hospital validation partition."""
        self.set_parameters(parameters)

        metrics, sample_count = evaluate_local(
            model=self.model,
            val_loader=self.val_loader,
            device=self.device
        )

        metrics["hospital_id"] = self.hospital_id
        clean_metrics = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k, v in metrics.items()}

        return float(metrics["loss"]), sample_count, clean_metrics


def create_client(hospital_id: str, partitions_root: Optional[Path] = None, batch_size: int = 32, lr: float = 0.001) -> StoneSenseFLClient:
    """Factory helper to instantiate a client for a given hospital."""
    if partitions_root is None:
        partitions_root = PROJECT_ROOT / "dl" / "datasets" / "partitions"
    hospital_dir = Path(partitions_root) / hospital_id
    return StoneSenseFLClient(hospital_id=hospital_id, partition_dir=hospital_dir, batch_size=batch_size, lr=lr)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Start StoneSense-AI Hospital Federated Client Node.")
    parser.add_argument("--hospital", type=str, required=True, choices=["hospital_1", "hospital_2", "hospital_3", "HOSP-001", "HOSP-002", "HOSP-003"], help="Hospital node ID")
    parser.add_argument("--server", type=str, default="127.0.0.1:8088", help="Flower server gRPC address")
    parser.add_argument("--batch-size", type=int, default=32, help="Local DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Local learning rate")

    args = parser.parse_args()

    code_map = {"HOSP-001": "hospital_1", "HOSP-002": "hospital_2", "HOSP-003": "hospital_3"}
    hospital_id = code_map.get(args.hospital, args.hospital)

    hospital_names = {
        "hospital_1": "Apollo Kidney Care (HOSP-001)",
        "hospital_2": "Manipal Urology Institute (HOSP-002)",
        "hospital_3": "AIIMS Nephrology Labs (HOSP-003)",
    }

    print("=" * 65)
    print(f"   STONESENSE-AI FEDERATED HOSPITAL NODE: {hospital_names.get(hospital_id, hospital_id).upper()}")
    print("=" * 65)
    print("Node Status:          CONNECTED & ISOLATED")
    print(f"Hospital Identifier:  {hospital_id} -> {hospital_names.get(hospital_id, hospital_id)}")
    print(f"Target Flower Server: {args.server}")

    partitions_root = PROJECT_ROOT / "dl" / "datasets" / "partitions"
    dist_file = partitions_root / "distribution.json"
    if dist_file.exists():
        try:
            with open(dist_file, "r", encoding="utf-8") as f:
                dist_data = json.load(f).get("distributions", {}).get(hospital_id, {})
                train_dist = dist_data.get("train", {})
                val_dist = dist_data.get("validation", {})
                print(f"Local Dataset Size:   {sum(train_dist.values()) + sum(val_dist.values())} CT scans")
                print(f"Class Distribution:   {train_dist}")
        except Exception:
            pass

    print("Privacy Protocol:     ZERO RAW CT DATA TRANSMISSION")
    print("                      [Raw patient CT scans remain on local disk]")
    print("                      [Only model weights & performance metrics are shared]")
    print("=" * 65)
    print(f"\n[{hospital_id}] Connecting to Flower Federated Server at {args.server}...")
    print(f"[{hospital_id}] {hospital_names.get(hospital_id, hospital_id)} connected. Waiting for server to start round...\n")

    client = create_client(hospital_id=hospital_id, partitions_root=partitions_root, batch_size=args.batch_size, lr=args.lr)

    try:
        fl.client.start_numpy_client(server_address=args.server, client=client)
    except Exception as e:
        print(f"\n[Connection Error] Could not connect to Flower Server at {args.server}.")
        print("Please verify that Terminal 2 (Flower Server) is running first on port 8088:")
        print("  Command: dl\\.venv\\Scripts\\python dl/federated/server.py --rounds 3 --min-clients 3\n")
        print(f"Details: {e}")