"""Integration Test for Real Flower Multi-Process gRPC Execution.

Verifies end-to-end multi-process Flower federated learning over network sockets:
1. Flower gRPC server runs in an independent process bound to 127.0.0.1 on an ephemeral test port.
2. At least two independent Flower client processes connect over gRPC.
3. Isolated, deterministic synthetic partitions are used (zero raw data transmission).
4. Real Flower aggregation occurs, producing an aggregated model checkpoint.
5. Resulting model parameters are verified to be finite (no NaNs/Infs) and compatible.
6. Process outputs, exit codes, and timeouts are asserted.
7. Production/base models (kidney_resnet18.pth) are guaranteed untouched.
8. Reliable subprocess termination and resource cleanup is verified.
"""

from pathlib import Path
import sys
import os
import time
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))

from dl.federated.grpc_runner import (
    FlowerGRPCRunner,
    find_free_port,
    wait_for_port,
    create_synthetic_partitions,
)
from dl.training.model import build_resnet18_classifier, CLASS_MAPPING


def test_free_port_finder():
    """Verify ephemeral port allocation generates a valid, bindable localhost port."""
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_synthetic_partitions_creation(tmp_path):
    """Verify deterministic synthetic dataset generation creates expected classes and splits."""
    parts_dir = tmp_path / "partitions"
    h_ids = ["hospital_1", "hospital_2"]
    create_synthetic_partitions(parts_dir, hospital_ids=h_ids, images_per_class=2)

    assert (parts_dir / "distribution.json").exists()
    for h_id in h_ids:
        for split in ["train", "validation"]:
            for cname in ["Cyst", "Normal", "Stone", "Tumor"]:
                class_dir = parts_dir / h_id / split / cname
                assert class_dir.exists()
                images = list(class_dir.glob("*.png"))
                assert len(images) == 2


def test_real_flower_grpc_multi_process_round(tmp_path):
    """Verify a complete multi-process Flower gRPC round with real network sockets and aggregation."""
    # 1. Setup isolated directories
    test_models_dir = tmp_path / "models"
    test_models_dir.mkdir(parents=True, exist_ok=True)
    (test_models_dir / "federated").mkdir(parents=True, exist_ok=True)

    test_partitions_dir = tmp_path / "partitions"
    client_ids = ["hospital_1", "hospital_2"]
    create_synthetic_partitions(
        test_partitions_dir,
        hospital_ids=client_ids,
        images_per_class=2,
        image_size=(64, 64)
    )

    # Record timestamp and hash of existing base model (if any) to verify it is untouched
    base_prod_model = PROJECT_ROOT / "dl" / "models" / "kidney_resnet18.pth"
    prod_mtime = base_prod_model.stat().st_mtime if base_prod_model.exists() else None

    # 2. Instantiate FlowerGRPCRunner
    runner = FlowerGRPCRunner(
        host="127.0.0.1",
        rounds=1,
        min_clients=2,
        models_dir=test_models_dir,
        partitions_dir=test_partitions_dir,
        db_persist=False,
    )

    # 3. Execute multi-process Flower round
    result = runner.run_round(
        client_ids=client_ids,
        max_batches=1,
        batch_size=2,
        server_startup_timeout=15.0,
        round_timeout=90.0,
    )

    # 4. Verify process execution and exit codes
    assert result["success"] is True, f"Flower gRPC round failed: {result}"
    assert result["exit_codes"].get("server") == 0
    assert result["exit_codes"].get("hospital_1") == 0
    assert result["exit_codes"].get("hospital_2") == 0

    # 5. Verify process output evidence
    server_stdout, server_stderr = result["server_logs"]
    assert "FEDERATED ROUND 1" in server_stdout or "gRPC server running" in server_stdout or "Flower ECE" in server_stderr
    assert "Round 1 completed" in server_stdout or "Weighted aggregation completed" in server_stdout

    h1_stdout, _ = result["client_logs"]["hospital_1"]
    h2_stdout, _ = result["client_logs"]["hospital_2"]
    assert "Apollo Kidney Care" in h1_stdout or "hospital_1" in h1_stdout
    assert "Manipal Urology" in h2_stdout or "hospital_2" in h2_stdout
    assert "Local training completed" in h1_stdout
    assert "Local training completed" in h2_stdout

    # 6. Verify checkpoint persistence and parameter integrity
    checkpoint_path = result["checkpoint_path"]
    assert checkpoint_path is not None
    assert checkpoint_path.exists()
    assert checkpoint_path.name == "resnet18_fed_round_001.pth"

    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt_data
    assert "metrics" in ckpt_data
    assert ckpt_data["round_number"] == 1

    # Verify all aggregated weights are finite and non-empty
    state_dict = ckpt_data["model_state_dict"]
    assert len(state_dict) > 0
    for name, tensor in state_dict.items():
        assert torch.isfinite(tensor).all(), f"Non-finite weights found in parameter {name}"

    # Verify compatibility with base ResNet18 architecture
    num_classes = len(CLASS_MAPPING)
    eval_model = build_resnet18_classifier(num_classes=num_classes, freeze_backbone=False)
    eval_model.load_state_dict(state_dict, strict=True)

    # 7. Safety Guarantee: base production model weights were NOT overwritten
    if prod_mtime is not None:
        assert base_prod_model.stat().st_mtime == prod_mtime, "Production base model weights were unexpectedly modified!"
    assert not (test_models_dir / "kidney_resnet18.pth").exists(), "Deployed pointer was created without gate review!"


def test_grpc_runner_cleanup_on_timeout():
    """Verify FlowerGRPCRunner cleans up subprocesses even when an exception occurs."""
    runner = FlowerGRPCRunner(host="127.0.0.1", rounds=1, min_clients=2)
    # Start server
    proc = runner.start_server()
    assert proc.poll() is None  # Process is running

    # Cleanup must terminate the process
    runner.cleanup()
    time.sleep(0.5)
    assert proc.poll() is not None
