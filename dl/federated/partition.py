"""StoneSense-AI Dataset Partitioning Module for Federated Learning.

Splits the processed CT dataset into 3 simulated hospital partitions.
Supports both IID and Non-IID distributions with reproducible seeding.
Saves distribution statistics to JSON.
"""

import os
import shutil
import random
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StoneSensePartitioner")

HOSPITAL_IDS = ["hospital_1", "hospital_2", "hospital_3"]
HOSPITAL_NAMES = {
    "hospital_1": "Apollo Kidney Care",
    "hospital_2": "Manipal Urology Institute",
    "hospital_3": "AIIMS Nephrology Labs",
}
CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]


def partition_dataset(
    source_dir: Path,
    output_dir: Path,
    mode: str = "iid",
    seed: int = 42,
    num_hospitals: int = 3
) -> Dict[str, Any]:
    """Partitions images from source processed directory into isolated hospital directories.

    Args:
        source_dir: Path to dl/processed containing train/validation/test.
        output_dir: Path to dl/datasets/partitions where hospital_1/2/3 will be stored.
        mode: "iid" or "non-iid".
        seed: Random seed for reproducibility.
        num_hospitals: Number of simulated hospitals (default 3).

    Returns:
        Dictionary containing partition metadata and class distribution summaries.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    random.seed(seed)

    logger.info(f"Partitioning dataset from '{source_dir}' to '{output_dir}' [Mode: {mode.upper()}, Seed: {seed}]")

    # Clean existing partitions
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hospital_dirs = {}
    for h_id in HOSPITAL_IDS[:num_hospitals]:
        h_path = output_dir / h_id
        for split in ["train", "validation", "test"]:
            for cls_name in CLASSES:
                (h_path / split / cls_name).mkdir(parents=True, exist_ok=True)
        hospital_dirs[h_id] = h_path

    distribution_summary: Dict[str, Dict[str, Dict[str, int]]] = {
        h_id: {"train": {c: 0 for c in CLASSES}, "validation": {c: 0 for c in CLASSES}, "test": {c: 0 for c in CLASSES}}
        for h_id in HOSPITAL_IDS[:num_hospitals]
    }

    # Partition each split
    for split in ["train", "validation", "test"]:
        split_src = source_dir / split
        if not split_src.exists():
            logger.warning(f"Split path {split_src} not found, skipping...")
            continue

        for cls_name in CLASSES:
            cls_src = split_src / cls_name
            if not cls_src.exists():
                continue

            images = sorted(list(cls_src.glob("*.*")))
            random.shuffle(images)
            n_images = len(images)

            if n_images == 0:
                continue

            if mode.lower() == "iid" or split != "train":
                # Equal split among clients for IID train, and always equal for val/test
                split_size = n_images // num_hospitals
                for i, h_id in enumerate(HOSPITAL_IDS[:num_hospitals]):
                    start_idx = i * split_size
                    end_idx = n_images if i == num_hospitals - 1 else (i + 1) * split_size
                    client_images = images[start_idx:end_idx]

                    dst_dir = hospital_dirs[h_id] / split / cls_name
                    for img in client_images:
                        shutil.copy2(img, dst_dir / img.name)

                    distribution_summary[h_id][split][cls_name] = len(client_images)
            else:
                # Non-IID Mode for train set: Skewed distributions
                # Hospital 1: Heavy Cyst (60%) & Normal (40%)
                # Hospital 2: Heavy Stone (70%) & Normal (30%)
                # Hospital 3: Heavy Tumor (70%) & Cyst (30%)
                if cls_name == "Cyst":
                    ratios = [0.60, 0.10, 0.30]
                elif cls_name == "Stone":
                    ratios = [0.10, 0.70, 0.20]
                elif cls_name == "Tumor":
                    ratios = [0.15, 0.15, 0.70]
                else:  # Normal
                    ratios = [0.45, 0.35, 0.20]

                # Normalize ratios
                total_ratio = sum(ratios[:num_hospitals])
                ratios = [r / total_ratio for r in ratios[:num_hospitals]]

                start_idx = 0
                for i, h_id in enumerate(HOSPITAL_IDS[:num_hospitals]):
                    if i == num_hospitals - 1:
                        client_images = images[start_idx:]
                    else:
                        count = int(n_images * ratios[i])
                        client_images = images[start_idx:start_idx + count]
                        start_idx += count

                    dst_dir = hospital_dirs[h_id] / split / cls_name
                    for img in client_images:
                        shutil.copy2(img, dst_dir / img.name)

                    distribution_summary[h_id][split][cls_name] = len(client_images)

    metadata = {
        "mode": mode,
        "seed": seed,
        "num_hospitals": num_hospitals,
        "hospital_names": HOSPITAL_NAMES,
        "classes": CLASSES,
        "distributions": distribution_summary,
    }

    # Save summary report
    summary_file = output_dir / "distribution.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset successfully partitioned into {num_hospitals} hospital subsets. Metadata saved to {summary_file}")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Partition CT dataset for Federated Learning.")
    parser.add_argument("--mode", type=str, default="iid", choices=["iid", "non-iid"], help="Distribution mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--src", type=str, default="dl/processed", help="Processed dataset root")
    parser.add_argument("--dst", type=str, default="dl/datasets/partitions", help="Partitions destination directory")

    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    src_path = root_dir / args.src
    dst_path = root_dir / args.dst

    partition_dataset(source_dir=src_path, output_dir=dst_path, mode=args.mode, seed=args.seed)
