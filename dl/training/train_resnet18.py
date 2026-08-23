"""Phase 4A — ResNet18 Model Training Script for StoneSense-AI.

Trains pretrained ResNet18 on processed CT image dataset using DataLoaders from Phase 3.
Saves:
- dl/models/kidney_resnet18.pth
- dl/models/best_checkpoint.pth
- dl/models/training_history.json
"""

import sys
import random
from pathlib import Path
import json
import logging
from typing import Dict, List, Any
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))

from model import build_resnet18_classifier, CLASS_MAPPING
from dataloaders import create_dataloaders
from transforms import get_train_transforms, get_val_test_transforms
from evaluate_helpers import evaluate_model, save_evaluation_charts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Phase4ATrainer")


def run_phase_4a_training(
    processed_dir: Path,
    models_dir: Path,
    batch_size: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
    patience: int = 3
) -> None:
    """Executes Phase 4A training pipeline."""
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Phase 4A Training initialized. Using device: {device}")

    class_names = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]

    # Load DataLoaders
    train_tf = get_train_transforms(image_size=(224, 224))
    val_tf = get_val_test_transforms(image_size=(224, 224))

    dataloaders = create_dataloaders(
        processed_dir=processed_dir,
        train_transform=train_tf,
        val_test_transform=val_tf,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["validation"]

    model = build_resnet18_classifier(num_classes=len(class_names), freeze_backbone=True, unfreeze_layer4=True)
    model.to(device)

    train_targets = [target for _, target in train_loader.dataset.samples]
    class_counts = torch.bincount(torch.tensor(train_targets), minlength=len(class_names)).float()
    class_weights = class_counts.sum() / (len(class_names) * class_counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": []
    }

    best_model_path = models_dir / "kidney_resnet18.pth"
    best_checkpoint_path = models_dir / "best_checkpoint.pth"

    logger.info("Starting Phase 4A training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
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

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Loss & Metrics
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss_sum += loss.item() * images.size(0)
                val_total += targets.size(0)

        val_metrics, _, _, _ = evaluate_model(model, val_loader, device, class_names)
        epoch_val_loss = val_loss_sum / val_total
        epoch_val_acc = val_metrics["accuracy"]
        epoch_val_f1 = val_metrics["f1_macro"]

        scheduler.step(epoch_val_f1)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["val_f1_macro"].append(round(epoch_val_f1, 4))

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                    f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val F1: {epoch_val_f1:.4f}")

        # Checkpoint handling
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_epoch = epoch
            patience_counter = 0

            logger.info(f"--> Improved Val Macro F1 ({best_val_f1:.4f}). Saving checkpoints...")
            torch.save(model.state_dict(), best_model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1_macro': best_val_f1,
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            logger.info(f"No improvement in Val F1 for {patience_counter} consecutive epoch(s).")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}!")
                break

    # Save training history
    history_path = models_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    logger.info(f"Phase 4A Complete! Best checkpoint saved from epoch {best_epoch} with Val F1: {best_val_f1:.4f}")


def main():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "processed"
    models_dir = base_dir / "models"

    run_phase_4a_training(
        processed_dir=processed_dir,
        models_dir=models_dir,
        batch_size=64,
        lr=0.001,
        epochs=2,
        patience=2
    )


if __name__ == "__main__":
    main()
