"""StoneSense-AI Flower Federated Learning Server.

Launches the Flower gRPC aggregation server for multi-hospital decentralized training.
Coordinates FedAvg aggregation across independent hospital client processes.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import sys
import argparse
import logging
import torch

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters
except ImportError:
    fl = None
    ndarrays_to_parameters = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

from model import build_resnet18_classifier, CLASS_MAPPING
from local_training import get_model_parameters
from strategy import StoneSenseFedAvg
from model_manager import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("StoneSenseFLServer")


def print_server_banner(
    num_classes: int,
    trainable_params: int,
    total_params: int,
    model_version: str,
    rounds: int,
    min_clients: int,
    server_address: str
):
    """Prints the comprehensive technical startup banner for StoneSense FL Server."""
    class_list = ", ".join([CLASS_MAPPING[i] for i in range(num_classes)])
    print("=" * 65)
    print("   STONESENSE-AI FEDERATED LEARNING SERVER (Flower gRPC)")
    print("=" * 65)
    print("")
    print("Base Model")
    print("-----------")
    print("Architecture:         ResNet18")
    print(f"Number of classes:    {num_classes} ({class_list})")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Initial model version:{model_version}")
    print("")
    print("Federated Learning Configuration")
    print("--------------------------------")
    print("Strategy:             StoneSenseFedAvg (Sample-Weighted FedAvg)")
    print(f"Total Target Rounds:  {rounds}")
    print(f"Min Clients Required: {min_clients}")
    print(f"gRPC Server Address:  {server_address}")
    print("Privacy Guarantee:    ZERO RAW CT DATA TRANSMISSION (Weights Only)")
    print("=" * 65)
    print("")
    print("[Flower Server] Federated Learning Server started.")
    print("[Flower Server] Base model: ResNet18")
    print(f"[Flower Server] Minimum clients required: {min_clients}")
    print(f"[Flower Server] Strategy: StoneSenseFedAvg")
    print("[Flower Server] Server waiting for hospital clients to connect...")
    print("")


def start_fl_server(
    server_address: str = "0.0.0.0:8088",
    rounds: int = 3,
    min_clients: int = 3,
    mode: str = "iid"
):
    if fl is None:
        raise RuntimeError("Flower (flwr) package is not installed in current environment.")

    # 1. Instantiate Base ResNet18 model and compute exact parameters
    num_classes = len(CLASS_MAPPING)
    base_model = build_resnet18_classifier(
        num_classes=num_classes,
        freeze_backbone=True,
        unfreeze_layer4=True
    )

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    # Check for existing checkpoint or default to v1
    latest_ckpt = model_manager.get_latest_checkpoint_path()
    initial_version = "resnet18_centralized_v1"
    if latest_ckpt and latest_ckpt.exists():
        try:
            ckpt_data = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                base_model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
                initial_version = ckpt_data.get("version_tag", initial_version)
        except Exception:
            pass

    # Extract initial weights
    initial_weights = get_model_parameters(base_model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # 2. Print technical banner
    print_server_banner(
        num_classes=num_classes,
        trainable_params=trainable_params,
        total_params=total_params,
        model_version=initial_version,
        rounds=rounds,
        min_clients=min_clients,
        server_address=server_address
    )

    # 3. Configure FedAvg Strategy
    strategy = StoneSenseFedAvg(
        num_classes=num_classes,
        mode=mode,
        db_persist=True,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
    )

    # 4. Start Flower gRPC Server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start StoneSense-AI Flower Federated Server.")
    parser.add_argument("--address", type=str, default="0.0.0.0:8088", help="Server gRPC address")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Minimum hospital clients required")
    parser.add_argument("--mode", type=str, default="iid", choices=["iid", "non-iid"], help="Partition distribution mode")

    args = parser.parse_args()
    start_fl_server(
        server_address=args.address,
        rounds=args.rounds,
        min_clients=args.min_clients,
        mode=args.mode
    )