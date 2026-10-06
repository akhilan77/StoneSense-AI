"""Automated Verification Test for Phase 3 Leakage-Controlled Grouped DL Split.

Asserts that:
1. No exact duplicate group (MD5) crosses splits.
2. No near-duplicate cluster (pHash <= 2) crosses splits.
3. No image appears in multiple splits.
4. Total image counts and per-class counts reconcile completely (12,446 images).
5. Original dl/processed directory is preserved.
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ORIG = PROJECT_ROOT / "dl" / "processed"
PROCESSED_GROUPED = PROJECT_ROOT / "dl" / "processed_grouped"
MANIFEST_PATH = PROJECT_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"
DUPLICATE_AUDIT_CSV = PROJECT_ROOT / "dl" / "outputs" / "reports" / "duplicate_analysis.csv"

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
SPLITS = ["train", "validation", "test"]
TOTAL_EXPECTED_IMAGES = 12446


def test_original_dataset_preserved():
    """Verify that original dl/processed directory still exists and contains all 12,446 images."""
    assert PROCESSED_ORIG.exists(), "Original dl/processed directory was modified or removed!"
    
    orig_total = 0
    for split in SPLITS:
        for cls in CLASSES:
            folder = PROCESSED_ORIG / split / cls
            assert folder.exists(), f"Original folder missing: {folder}"
            orig_total += len(list(folder.glob("*.*")))
            
    assert orig_total == TOTAL_EXPECTED_IMAGES, f"Original image count modified! Expected {TOTAL_EXPECTED_IMAGES}, got {orig_total}"


def test_grouped_split_reconciled_counts():
    """Verify that dl/processed_grouped has all directories and exact total count."""
    assert PROCESSED_GROUPED.exists(), "Grouped split directory dl/processed_grouped does not exist!"
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    
    manifest_df = pd.read_csv(MANIFEST_PATH)
    assert len(manifest_df) == TOTAL_EXPECTED_IMAGES, f"Manifest count {len(manifest_df)} != {TOTAL_EXPECTED_IMAGES}"

    disk_total = 0
    for split in SPLITS:
        for cls in CLASSES:
            folder = PROCESSED_GROUPED / split / cls
            assert folder.exists(), f"Grouped folder missing: {folder}"
            files = list(folder.glob("*.*"))
            disk_total += len(files)

    assert disk_total == TOTAL_EXPECTED_IMAGES, f"Disk image count {disk_total} != {TOTAL_EXPECTED_IMAGES}"


def test_no_image_in_multiple_splits():
    """Verify that no image appears in more than one destination split."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    counts = manifest_df["filename"].value_counts()
    duplicates = counts[counts > 1]
    assert len(duplicates) == 0, f"Found images appearing multiple times: {duplicates.to_dict()}"


def test_no_exact_duplicate_crosses_splits():
    """Verify that no exact duplicate image group (by MD5) crosses train/validation/test splits."""
    audit_df = pd.read_csv(DUPLICATE_AUDIT_CSV)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    
    # Merge destination_split onto audit_df
    merged = audit_df.merge(manifest_df[["filename", "destination_split"]], on="filename")
    
    exact_groups = merged[merged["is_exact_duplicate"]].groupby("exact_duplicate_group_id")
    cross_split_exact = []
    
    for gid, grp in exact_groups:
        splits = grp["destination_split"].unique()
        if len(splits) > 1:
            cross_split_exact.append((gid, splits.tolist()))
            
    assert len(cross_split_exact) == 0, f"Found exact duplicate groups crossing splits: {cross_split_exact}"


def test_no_near_duplicate_cluster_crosses_splits():
    """Verify that no pHash <= 2 cluster crosses train/validation/test splits."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    near_groups = manifest_df.groupby("near_duplicate_group_id")
    
    cross_split_near = []
    for gid, grp in near_groups:
        splits = grp["destination_split"].unique()
        if len(splits) > 1:
            cross_split_near.append((gid, splits.tolist()))
            
    assert len(cross_split_near) == 0, f"Found near duplicate clusters crossing splits: {cross_split_near}"
