"""StoneSense-AI Flower Federated Learning Client.

Implements flwr.client.NumPyClient for isolated hospital nodes.
Trains locally on hospital partition without exposing raw CT images.
"""

from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


class StoneSenseFLClient(fl.client.NumPyClient):
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

        logger.info(f"[{self.hospital_id}] Starting Round {current_round} local training ({local_epochs} epoch(s))...")

        self.model, metrics, sample_count = train_local(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=local_epochs,
            lr=lr,
            device=self.device,
            class_names=self.class_names
        )

        logger.info(
            f"[{self.hospital_id}] Round {current_round} Complete: "
            f"Train Acc: {metrics['train_accuracy']:.4f}, Val F1: {metrics.get('val_f1_macro', 0.0):.4f}"
        )

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


def create_client(hospital_id: str, partitions_root: Optional[Path] = None) -> StoneSenseFLClient:
    """Factory helper to instantiate a client for a given hospital."""
    if partitions_root is None:
        partitions_root = PROJECT_ROOT / "dl" / "datasets" / "partitions"
    hospital_dir = Path(partitions_root) / hospital_id
    return StoneSenseFLClient(hospital_id=hospital_id, partition_dir=hospital_dir)
