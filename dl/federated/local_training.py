"""StoneSense-AI Local Training Module for Federated Learning.

Provides isolated local training logic for individual hospital client partitions.
Converts model weights to/from NumPy parameter lists for Flower interoperability.
"""

from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("StoneSenseLocalTraining")


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extracts trainable model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Loads a list of NumPy arrays into the model state dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 1,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None
) -> Tuple[nn.Module, Dict[str, float], int]:
    """Trains model on local data loader and computes evaluation metrics.

    Args:
        model: PyTorch classification model.
        train_loader: DataLoader for local training partition.
        val_loader: Optional DataLoader for local validation partition.
        epochs: Number of local training epochs.
        lr: Learning rate for local Adam optimizer.
        device: PyTorch device (CPU or CUDA).
        class_names: Optional class names list.

    Returns:
        Tuple containing (trained_model, metrics_dict, sample_count).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    num_classes = 4
    if class_names is None:
        class_names = ["Cyst", "Normal", "Stone", "Tumor"]
    num_classes = len(class_names)

    # Compute class weights to handle non-IID class skew if needed
    train_targets = []
    for _, targets in train_loader:
        train_targets.extend(targets.tolist())

    sample_count = len(train_targets)
    if sample_count == 0:
        return model, {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0}, 0

    class_counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
    class_weights = class_counts.sum() / (num_classes * class_counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer on trainable parameters only
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    running_loss = 0.0
    all_train_preds = []
    all_train_targets = []

    for epoch in range(epochs):
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_targets.extend(targets.cpu().numpy())

    avg_train_loss = running_loss / (sample_count * epochs)
    train_acc = float(accuracy_score(all_train_targets, all_train_preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_train_targets, all_train_preds, average='macro', zero_division=0
    )

    metrics: Dict[str, float] = {
        "train_loss": float(avg_train_loss),
        "train_accuracy": float(train_acc),
        "train_precision_macro": float(p_macro),
        "train_recall_macro": float(r_macro),
        "train_f1_macro": float(f1_macro),
        "sample_count": float(sample_count),
    }

    # Evaluate on val_loader if provided
    if val_loader is not None and len(val_loader) > 0:
        val_metrics, val_samples = evaluate_local(model, val_loader, device=device, criterion=criterion)
        metrics.update({
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_sample_count": float(val_samples),
        })

    return model, metrics, sample_count


def evaluate_local(
    model: nn.Module,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None
) -> Tuple[Dict[str, float], int]:
    """Evaluates the model on local validation/test data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()

    val_loss_sum = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss_sum += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}, 0

    val_loss = val_loss_sum / total_samples
    acc = float(accuracy_score(all_targets, all_preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    return {
        "loss": float(val_loss),
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
    }, total_samples
