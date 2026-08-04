"""PyTorch Dataset & DataLoader utilities for CT Kidney Images (DL).

Implements CTKidneyDataset and create_dataloaders factory function.
"""

from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("CTDataLoaders")


class CTKidneyDataset(Dataset):
    """PyTorch Dataset for CT Kidney Images."""

    def __init__(
        self,
        split_dir: Path,
        class_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None
    ):
        self.split_dir = Path(split_dir)
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        
        # Detect classes if not provided
        classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        if class_to_idx is None:
            self.class_to_idx = {cname: idx for idx, cname in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.classes = list(self.class_to_idx.keys())

        for cname in self.classes:
            cdir = self.split_dir / cname
            if cdir.exists():
                c_idx = self.class_to_idx[cname]
                for img_path in cdir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append((img_path, c_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def create_dataloaders(
    processed_dir: Path,
    train_transform: Callable,
    val_test_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """Creates PyTorch DataLoaders for Train, Validation, and Test sets."""
    processed_dir = Path(processed_dir)
    
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"

    # Build class_to_idx mapping consistently from train_dir
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_to_idx = {cname: idx for idx, cname in enumerate(train_classes)}

    train_ds = CTKidneyDataset(train_dir, class_to_idx=class_to_idx, transform=train_transform)
    val_ds = CTKidneyDataset(val_dir, class_to_idx=class_to_idx, transform=val_test_transform)
    test_ds = CTKidneyDataset(test_dir, class_to_idx=class_to_idx, transform=val_test_transform)

    # Check CUDA availability for pin_memory
    use_pin = pin_memory and torch.cuda.is_available()

    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin
        ),
        "validation": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin
        )
    }

    logger.info(f"DataLoaders created - Train batches: {len(dataloaders['train'])}, "
                f"Val batches: {len(dataloaders['validation'])}, Test batches: {len(dataloaders['test'])}")
    return dataloaders
