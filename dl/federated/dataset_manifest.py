"""StoneSense-AI Hospital Local Dataset Manifest & Integrity Tracker.

Provides local-only manifest indexing, cryptographic hashing, patient-level grouping validation,
leakage detection, and delta change scanning for hospital partitions:
- Detects added, modified, removed, duplicate, and corrupted CT images
- Verifies image integrity without moving raw data
- Enforces patient/scan-level grouping to prevent train/val leakage
- Determines whether valid new training data exists before triggering retraining
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

logger = logging.getLogger("DatasetManifest")

DATASET_CLASSES = ("Cyst", "Normal", "Stone", "Tumor")
DATASET_SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of a local image file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_patient_group(filename: str) -> str:
    """Extracts clinical patient/case identifier from image filename to preserve grouping."""
    stem = Path(filename).stem
    # Match patterns like: case_0012_slice_05, patient_34_scan_1, p123_45, CT_009_02
    match = re.match(r"^([a-zA-Z]+_?\d+)", stem)
    if match:
        return match.group(1).lower()
    # Fallback to prefix before first hyphen/underscore
    parts = re.split(r"[-_]", stem)
    return parts[0].lower() if parts else stem.lower()


def scan_partition_dataset(
    partition_dir: Path,
    hospital_code: str = "HOSP-001",
    save_manifest: bool = True,
) -> Dict[str, Any]:
    """Scans hospital partition, verifies image integrity, detects changes, and enforces patient split safety."""
    p_dir = Path(partition_dir)
    manifest_path = p_dir / "manifest.json"

    # Load existing manifest if present
    previous_manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                previous_manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load previous manifest at {manifest_path}: {e}")

    prev_files: Dict[str, str] = previous_manifest.get("file_hashes", {})  # rel_path -> sha256
    prev_version = previous_manifest.get("version_number", 0)

    current_files: Dict[str, Dict[str, Any]] = {}
    file_hashes: Dict[str, str] = {}
    split_counts: Dict[str, int] = {s: 0 for s in DATASET_SPLITS}
    class_counts: Dict[str, int] = {c: 0 for c in DATASET_CLASSES}
    patients_by_split: Dict[str, Set[str]] = {s: set() for s in DATASET_SPLITS}

    hash_to_paths: Dict[str, List[str]] = {}
    corrupted_files: List[str] = []

    for split in DATASET_SPLITS:
        split_path = p_dir / split
        if not split_path.is_dir():
            continue

        for class_name in DATASET_CLASSES:
            class_path = split_path / class_name
            if not class_path.is_dir():
                continue

            for img_path in class_path.rglob("*"):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                rel_path = str(img_path.relative_to(p_dir)).replace("\\", "/")
                patient_id = extract_patient_group(img_path.name)

                # Verify image integrity
                is_corrupt = False
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    is_corrupt = True
                    corrupted_files.append(rel_path)

                if not is_corrupt:
                    file_sha = compute_file_sha256(img_path)
                    file_size = img_path.stat().st_size

                    file_hashes[rel_path] = file_sha
                    current_files[rel_path] = {
                        "split": split,
                        "class": class_name,
                        "sha256": file_sha,
                        "size_bytes": file_size,
                        "patient_id": patient_id,
                    }
                    split_counts[split] += 1
                    class_counts[class_name] += 1
                    patients_by_split[split].add(patient_id)

                    hash_to_paths.setdefault(file_sha, []).append(rel_path)

    # Detect duplicate files
    duplicate_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    duplicate_count = sum(len(paths) - 1 for paths in duplicate_groups.values())

    # Detect patient leakage across splits
    train_patients = patients_by_split.get("train", set())
    val_patients = patients_by_split.get("validation", set())
    test_patients = patients_by_split.get("test", set())

    leakage_train_val = train_patients.intersection(val_patients)
    leakage_train_test = train_patients.intersection(test_patients)
    patient_leakage_detected = bool(leakage_train_val or leakage_train_test)

    # Compute delta vs previous manifest
    added_files = [p for p in file_hashes if p not in prev_files]
    removed_files = [p for p in prev_files if p not in file_hashes]
    modified_files = [p for p in file_hashes if p in prev_files and file_hashes[p] != prev_files[p]]
    unchanged_files = [p for p in file_hashes if p in prev_files and file_hashes[p] == prev_files[p]]

    new_train_images = sum(1 for p in (added_files + modified_files) if current_files.get(p, {}).get("split") == "train")
    has_new_valid_training_data = bool(
        new_train_images > 0
        and len(corrupted_files) == 0
        and not patient_leakage_detected
    )

    new_version_number = prev_version + 1 if (added_files or modified_files or removed_files) else max(prev_version, 1)
    version_tag = f"{hospital_code}_manifest_v{new_version_number:03d}"

    manifest_data = {
        "hospital_code": hospital_code,
        "version_tag": version_tag,
        "version_number": new_version_number,
        "scanned_at": datetime.utcnow().isoformat(),
        "total_images": sum(split_counts.values()),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "unique_patients_by_split": {s: len(pats) for s, pats in patients_by_split.items()},
        "patient_leakage_detected": patient_leakage_detected,
        "leaked_patient_count": len(leakage_train_val) + len(leakage_train_test),
        "corrupted_count": len(corrupted_files),
        "corrupted_files": corrupted_files[:10],
        "duplicate_count": duplicate_count,
        "has_new_valid_training_data": has_new_valid_training_data,
        "delta": {
            "added_count": len(added_files),
            "modified_count": len(modified_files),
            "removed_count": len(removed_files),
            "unchanged_count": len(unchanged_files),
            "new_train_images": new_train_images,
        },
        "file_hashes": file_hashes,
    }

    if save_manifest and p_dir.exists():
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            logger.info(f"Saved dataset manifest for {hospital_code} ({version_tag}) at {manifest_path}")
        except Exception as e:
            logger.warning(f"Could not write manifest to {manifest_path}: {e}")

    return manifest_data
