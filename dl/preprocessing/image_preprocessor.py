"""Main Deep Learning Image Preprocessing Script for CT Kidney Dataset.

Splits dataset (70/15/15), initializes PyTorch DataLoaders, verifies sample batches,
saves dl/artifacts/image_metadata.json, and writes preprocessing_report.md.
"""

from pathlib import Path
import json
import logging
import torch

from transforms import get_train_transforms, get_val_test_transforms, IMAGENET_MEAN, IMAGENET_STD
from split_dataset import split_ct_dataset
from dataloaders import create_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DLImagePreprocessor")


def run_dl_preprocessing_pipeline() -> None:
    """Executes end-to-end DL image preprocessing pipeline."""
    base_dir = Path(__file__).resolve().parents[1]
    dataset_dir = base_dir / "datasets" / "archive" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    processed_dir = base_dir / "processed"
    artifacts_dir = base_dir / "artifacts"
    reports_dir = base_dir / "outputs" / "reports"

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 3 Deep Learning Image Preprocessing Pipeline...")

    # 1. Stratified Dataset Split & File Copy
    split_counts = split_ct_dataset(dataset_dir=dataset_dir, output_dir=processed_dir, random_state=42)

    # 2. Get Transformation Pipelines
    train_tf = get_train_transforms(image_size=(224, 224))
    val_tf = get_val_test_transforms(image_size=(224, 224))

    # 3. Create PyTorch DataLoaders
    dataloaders = create_dataloaders(
        processed_dir=processed_dir,
        train_transform=train_tf,
        val_test_transform=val_tf,
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )

    # 4. Verify DataLoader Sample Batch
    train_loader = dataloaders["train"]
    sample_images, sample_labels = next(iter(train_loader))
    logger.info(f"Verified sample batch - Images shape: {sample_images.shape}, Labels shape: {sample_labels.shape}")

    # 5. Extract metadata & generate image_metadata.json
    classes = sorted(list(split_counts["train"].keys()))
    total_images = sum(sum(c.values()) for c in split_counts.values())

    metadata = {
        "dataset_name": "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
        "classes": classes,
        "total_images": total_images,
        "image_dimensions": [224, 224, 3],
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "transform_pipeline": {
            "train": [
                "Resize(224, 224)",
                "RandomHorizontalFlip(p=0.5)",
                "RandomRotation(15)",
                "ColorJitter(brightness=0.1, contrast=0.1)",
                "ToTensor()",
                "Normalize(ImageNet)"
            ],
            "validation_test": [
                "Resize(224, 224)",
                "ToTensor()",
                "Normalize(ImageNet)"
            ]
        },
        "split_counts": split_counts,
        "batch_size": 32
    }

    metadata_path = artifacts_dir / "image_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved DL image metadata JSON to {metadata_path}")

    # 6. Generate preprocessing_report.md
    train_sum = sum(split_counts['train'].values())
    val_sum = sum(split_counts['validation'].values())
    test_sum = sum(split_counts['test'].values())

    report_md = f"""# CT Kidney Image Preprocessing & DataLoader Report

**Dataset Directory:** `{dataset_dir}`  
**Status:** Completed  

---

## 📊 Dataset Stratified Split Statistics (70% / 15% / 15%)

| Split | Total Images | Cyst | Normal | Stone | Tumor |
| --- | --- | --- | --- | --- | --- |
| **Train (70%)** | {train_sum} | {split_counts['train'].get('Cyst', 0)} | {split_counts['train'].get('Normal', 0)} | {split_counts['train'].get('Stone', 0)} | {split_counts['train'].get('Tumor', 0)} |
| **Validation (15%)** | {val_sum} | {split_counts['validation'].get('Cyst', 0)} | {split_counts['validation'].get('Normal', 0)} | {split_counts['validation'].get('Stone', 0)} | {split_counts['validation'].get('Tumor', 0)} |
| **Test (15%)** | {test_sum} | {split_counts['test'].get('Cyst', 0)} | {split_counts['test'].get('Normal', 0)} | {split_counts['test'].get('Stone', 0)} | {split_counts['test'].get('Tumor', 0)} |

---

## ⚙️ Preprocessing & Augmentation Strategy

- **Input Dimension:** Resized to standard `224 x 224` pixels.
- **Normalization:** Standard ImageNet Mean (`[0.485, 0.456, 0.406]`) and Standard Deviation (`[0.229, 0.224, 0.225]`).
- **Data Augmentations (Train Only):**
  - Random Horizontal Flips (p=0.5)
  - Random Rotations (+/- 15 deg)
  - Color Jitter (Brightness/Contrast +/- 10%)

---

## 📁 Saved Artifacts & Verified Batches

- **Metadata JSON:** `dl/artifacts/image_metadata.json`
- **Processed Split Directories:** `dl/processed/train/`, `dl/processed/validation/`, `dl/processed/test/`
- **Sample DataLoader Batch Size:** `32` images per batch (`Batch Shape: [32, 3, 224, 224]`).

---

## 🚀 Recommendations for Phase 4 Model Training

1. **Backbone Architecture:** Use Transfer Learning with `ResNet18` pre-trained on ImageNet.
2. **Loss Function:** `CrossEntropyLoss` with class weights to account for class imbalance (e.g. `Stone` class).
3. **Optimizer:** AdamW optimizer with initial learning rate `1e-4` and cosine annealing learning rate scheduler.
"""

    report_path = reports_dir / "preprocessing_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Saved preprocessing report to {report_path}")
    logger.info("DL Image Preprocessing Pipeline executed successfully.")


if __name__ == "__main__":
    run_dl_preprocessing_pipeline()
