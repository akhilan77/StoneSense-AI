"""Unit tests for dataset partitioning module."""

from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(PROJECT_ROOT / "dl" / "federated"))

from partition import partition_dataset


def test_partition_dataset_iid(tmp_path):
    # Create mock dataset
    source_dir = tmp_path / "mock_processed"
    for split in ["train", "validation", "test"]:
        for cls_name in ["Cyst", "Normal", "Stone", "Tumor"]:
            c_dir = source_dir / split / cls_name
            c_dir.mkdir(parents=True, exist_ok=True)
            for i in range(6):
                (c_dir / f"img_{i}.jpg").write_text("mock image data")

    out_dir = tmp_path / "mock_partitions"
    meta = partition_dataset(source_dir=source_dir, output_dir=out_dir, mode="iid", seed=42)

    assert out_dir.exists()
    assert (out_dir / "distribution.json").exists()
    assert len(meta["distributions"]) == 3

    for h_id in ["hospital_1", "hospital_2", "hospital_3"]:
        assert (out_dir / h_id).exists()
        # Each hospital should have 2 images per class in train (6 total / 3)
        assert meta["distributions"][h_id]["train"]["Cyst"] == 2
        assert meta["distributions"][h_id]["train"]["Stone"] == 2
