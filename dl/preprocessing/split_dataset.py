"""DL Dataset Splitting Module for CT Kidney Images.

Performs stratified splitting (70% Train, 15% Validation, 15% Test) on image files
and constructs split directory structures under dl/processed/.
"""

import shutil
import random
from collections import defaultdict
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("DLSplitter")


def split_ct_dataset(
    dataset_dir: Path,
    output_dir: Path,
    random_state: int = 42
) -> Dict[str, Dict[str, int]]:
    """Splits CT images into train/validation/test directories with stratification using pure Python."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Locate class directories
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name == dataset_dir.name:
        subdirs = [d for d in subdirs[0].iterdir() if d.is_dir()]

    class_files: Dict[str, List[Path]] = defaultdict(list)
    for d in subdirs:
        cname = d.name.capitalize()
        images = [
            f for f in d.rglob("*")
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
        class_files[cname] = images

    random.seed(random_state)

    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for cname, files in class_files.items():
        shuffled = files.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)

        c_train = shuffled[:n_train]
        c_val = shuffled[n_train:n_train + n_val]
        c_test = shuffled[n_train + n_val:]

        train_paths.extend(c_train)
        train_labels.extend([cname] * len(c_train))

        val_paths.extend(c_val)
        val_labels.extend([cname] * len(c_val))

        test_paths.extend(c_test)
        test_labels.extend([cname] * len(c_test))

    splits = {
        "train": (train_paths, train_labels),
        "validation": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }

    split_counts: Dict[str, Dict[str, int]] = {}

    for split_name, (paths, labels) in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        counts: Dict[str, int] = {}

        for src_path, label in zip(paths, labels):
            dst_class_dir = split_dir / label
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            dst_file = dst_class_dir / src_path.name

            if not dst_file.exists():
                shutil.copy2(src_path, dst_file)

            counts[label] = counts.get(label, 0) + 1

        split_counts[split_name] = counts
        logger.info(f"Split '{split_name}' prepared with {len(paths)} images: {counts}")

    return split_counts
