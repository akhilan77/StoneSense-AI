"""Integration checks for the materialized duplicate-aware CT split."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_GROUPED = PROJECT_ROOT / "dl" / "processed_grouped"
REPORTS_DIR = PROJECT_ROOT / "dl" / "outputs" / "reports"
MANIFEST_PATH = REPORTS_DIR / "grouped_split_manifest.csv"
DUPLICATE_AUDIT_CSV = REPORTS_DIR / "duplicate_analysis.csv"
PHASH_CLUSTERS_JSON = REPORTS_DIR / "phash_clusters.json"

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
SPLITS = ["train", "validation", "test"]
TOTAL_EXPECTED_IMAGES = 12446


def test_original_dataset_preserved():
    """Verify that source images remain intact at their manifest-recorded paths."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    manifest_df = pd.read_csv(MANIFEST_PATH)
    assert manifest_df["source_path"].map(lambda path: Path(path).is_file()).all()
    assert manifest_df["source_path"].nunique() == TOTAL_EXPECTED_IMAGES


def test_grouped_split_reconciled_counts():
    """Verify that every manifest image was copied exactly once to the split tree."""
    assert PROCESSED_GROUPED.exists(), "Grouped split directory is missing!"
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    manifest_df = pd.read_csv(MANIFEST_PATH)
    assert len(manifest_df) == TOTAL_EXPECTED_IMAGES
    assert set(manifest_df["class"]) == set(CLASSES)
    assert set(manifest_df["destination_split"]) == set(SPLITS)
    assert manifest_df["destination_path"].map(lambda path: Path(path).is_file()).all()

    for split in SPLITS:
        for class_name in CLASSES:
            folder = PROCESSED_GROUPED / split / class_name
            assert folder.is_dir(), f"Grouped folder missing: {folder}"
            assert any(path.is_file() for path in folder.iterdir()), f"No images in {folder}"
    assert sum(
        path.is_file()
        for path in PROCESSED_GROUPED.rglob("*")
    ) == TOTAL_EXPECTED_IMAGES


def test_no_image_in_multiple_splits():
    """Verify unique file identities and exactly one split per source path."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    assert manifest_df["filename"].is_unique
    assert manifest_df["source_path"].is_unique
    assert manifest_df.groupby("source_path")["destination_split"].nunique().max() == 1


def test_no_exact_duplicate_crosses_splits():
    """Verify SHA-256 exact duplicate groups stay within one split/component."""
    audit_df = pd.read_csv(DUPLICATE_AUDIT_CSV)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    assert {
        "filename",
        "sha256",
        "is_exact_duplicate",
        "exact_duplicate_group_id",
    }.issubset(audit_df.columns)
    assert audit_df["filename"].is_unique
    merged = audit_df.merge(
        manifest_df[["filename", "destination_split"]],
        on="filename",
        validate="one_to_one",
    )
    assert len(merged) == TOTAL_EXPECTED_IMAGES
    exact_groups = merged[merged["is_exact_duplicate"]].groupby(
        "exact_duplicate_group_id"
    )
    for _, group in exact_groups:
        assert group["destination_split"].nunique() == 1
        assert group["near_duplicate_group_id"].nunique() == 1


def test_no_near_duplicate_cluster_crosses_splits():
    """Verify pHash-distance<=2 connected components never cross splits."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    assert manifest_df["near_duplicate_group_id"].notna().all()
    assert (
        manifest_df.groupby("near_duplicate_group_id")["destination_split"]
        .nunique()
        .max()
        == 1
    )


def test_class_coverage_and_p_hash_report():
    assert PHASH_CLUSTERS_JSON.exists()
    manifest_df = pd.read_csv(MANIFEST_PATH)
    counts = pd.crosstab(manifest_df["destination_split"], manifest_df["class"])
    assert set(counts.index) == set(SPLITS)
    assert set(counts.columns) == set(CLASSES)
    assert (counts > 0).all().all()
