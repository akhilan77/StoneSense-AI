"""Automated Leakage and Isolation Verification Test for Partitioned Hospital Datasets.

Asserts that:
1. Every individual hospital partition (hospital_1, hospital_2, hospital_3) has 0 MD5 overlap across train, validation, and test.
2. Across all hospitals pooled, train vs validation vs test has 0 MD5 overlap.
3. No image file is duplicated across hospitals within any split (strictly disjoint partitioning).
4. No near-duplicate cluster / grouped patient unit crosses train, validation, or test partitions.
5. Total image counts and per-class counts reconcile completely against dl/processed_grouped (8,708 / 1,869 / 1,869 = 12,446).
"""

import hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTITIONS_DIR = PROJECT_ROOT / "dl" / "datasets" / "partitions"
PROCESSED_GROUPED = PROJECT_ROOT / "dl" / "processed_grouped"
MANIFEST_PATH = PROJECT_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"

HOSPITALS = ["hospital_1", "hospital_2", "hospital_3"]
SPLITS = ["train", "validation", "test"]
CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]

EXPECTED_COUNTS = {
    "train": {"Cyst": 2595, "Normal": 3553, "Stone": 963, "Tumor": 1597, "total": 8708},
    "validation": {"Cyst": 557, "Normal": 762, "Stone": 207, "Tumor": 343, "total": 1869},
    "test": {"Cyst": 557, "Normal": 762, "Stone": 207, "Tumor": 343, "total": 1869},
}


def _compute_file_md5(filepath: Path) -> str:
    """Computes MD5 hash of a file on disk."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@pytest.fixture(scope="module")
def partition_file_cache():
    """Caches partition file metadata and MD5s for high-speed test execution."""
    cache = {}
    for h in HOSPITALS:
        for split in SPLITS:
            for c in CLASSES:
                folder = PARTITIONS_DIR / h / split / c
                if folder.exists():
                    for img in folder.glob("*.*"):
                        cache[img.resolve()] = {
                            "hospital": h,
                            "split": split,
                            "class": c,
                            "filename": img.name,
                            "md5": _compute_file_md5(img),
                        }
    return cache


def test_partition_counts_reconcile_with_grouped_source(partition_file_cache):
    """Verify that partitioned image counts exactly match grouped dataset counts per split and class."""
    assert PARTITIONS_DIR.exists(), "Partitions directory does not exist!"

    for split in SPLITS:
        pooled_class_counts = {c: 0 for c in CLASSES}
        for item in partition_file_cache.values():
            if item["split"] == split:
                pooled_class_counts[item["class"]] += 1

        expected = EXPECTED_COUNTS[split]
        for c in CLASSES:
            assert pooled_class_counts[c] == expected[c], (
                f"Mismatch in {split} {c}: got {pooled_class_counts[c]}, expected {expected[c]}"
            )
        assert sum(pooled_class_counts.values()) == expected["total"]


def test_per_hospital_zero_md5_overlap_across_splits(partition_file_cache):
    """Verify each individual hospital has zero MD5 hash overlap across train, val, and test."""
    for h in HOSPITALS:
        split_hashes = defaultdict(set)
        for item in partition_file_cache.values():
            if item["hospital"] == h:
                split_hashes[item["split"]].add(item["md5"])

        # Assert pairwise disjoint across splits
        tv_overlap = split_hashes["train"] & split_hashes["validation"]
        tt_overlap = split_hashes["train"] & split_hashes["test"]
        vt_overlap = split_hashes["validation"] & split_hashes["test"]

        assert len(tv_overlap) == 0, f"{h} has {len(tv_overlap)} MD5 overlap between train and validation"
        assert len(tt_overlap) == 0, f"{h} has {len(tt_overlap)} MD5 overlap between train and test"
        assert len(vt_overlap) == 0, f"{h} has {len(vt_overlap)} MD5 overlap between validation and test"


def test_pooled_cross_hospital_zero_md5_overlap(partition_file_cache):
    """Verify pooled global partitions across all hospitals have zero MD5 overlap across splits."""
    pooled_split_hashes = defaultdict(set)
    for item in partition_file_cache.values():
        pooled_split_hashes[item["split"]].add(item["md5"])

    tv_overlap = pooled_split_hashes["train"] & pooled_split_hashes["validation"]
    tt_overlap = pooled_split_hashes["train"] & pooled_split_hashes["test"]
    vt_overlap = pooled_split_hashes["validation"] & pooled_split_hashes["test"]

    assert len(tv_overlap) == 0, f"Pooled partitions have {len(tv_overlap)} train-val MD5 overlap"
    assert len(tt_overlap) == 0, f"Pooled partitions have {len(tt_overlap)} train-test MD5 overlap"
    assert len(vt_overlap) == 0, f"Pooled partitions have {len(vt_overlap)} val-test MD5 overlap"


def test_no_image_duplicated_across_hospitals_within_same_split(partition_file_cache):
    """Verify no image file is assigned to multiple hospitals within the same split."""
    for split in SPLITS:
        seen_filenames = set()
        for item in partition_file_cache.values():
            if item["split"] == split:
                fname = item["filename"]
                assert fname not in seen_filenames, f"Duplicate filename {fname} across hospitals in {split}"
                seen_filenames.add(fname)


def test_no_near_duplicate_cluster_crosses_partition_splits(partition_file_cache):
    """Verify that no perceptual hash (pHash <= 2) cluster crosses train, val, and test splits."""
    assert MANIFEST_PATH.exists(), f"Manifest missing: {MANIFEST_PATH}"
    manifest_df = pd.read_csv(MANIFEST_PATH)
    file_to_cluster = dict(zip(manifest_df["filename"], manifest_df["near_duplicate_group_id"]))

    split_clusters = defaultdict(set)
    for item in partition_file_cache.values():
        cluster_id = file_to_cluster.get(item["filename"])
        if cluster_id is not None:
            split_clusters[item["split"]].add(cluster_id)

    tv_clusters = split_clusters["train"] & split_clusters["validation"]
    tt_clusters = split_clusters["train"] & split_clusters["test"]
    vt_clusters = split_clusters["validation"] & split_clusters["test"]

    assert len(tv_clusters) == 0, f"Found {len(tv_clusters)} near-duplicate clusters crossing train-val partitions"
    assert len(tt_clusters) == 0, f"Found {len(tt_clusters)} near-duplicate clusters crossing train-test partitions"
    assert len(vt_clusters) == 0, f"Found {len(vt_clusters)} near-duplicate clusters crossing val-test partitions"
