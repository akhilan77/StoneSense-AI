"""Unit tests for local isolated training module."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

from local_training import train_local, evaluate_local, get_model_parameters, set_model_parameters


class MockClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.fc(x)


def test_train_local_and_parameter_sync():
    torch.manual_seed(42)
    model = MockClassifier(num_classes=4)

    # Mock synthetic features and targets
    x = torch.randn(32, 10)
    y = torch.randint(0, 4, (32,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    # Test parameter extraction
    params = get_model_parameters(model)
    assert len(params) == 2  # weight and bias
    assert params[0].shape == (4, 10)

    # Test local training step
    trained_model, metrics, samples = train_local(
        model=model,
        train_loader=loader,
        val_loader=loader,
        epochs=1,
        lr=0.01,
        device=torch.device("cpu")
    )

    assert samples == 32
    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert "train_f1_macro" in metrics
    assert "val_accuracy" in metrics

    # Test parameter restoration
    updated_params = get_model_parameters(trained_model)
    set_model_parameters(model, updated_params)
