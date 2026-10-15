"""Unit tests for deterministic exact/perceptual grouping and grouped splits."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from dl.preprocessing.grouping import (
    CLASS_NAMES,
    ImageRecord,
    assign_group_splits,
    build_components,
    build_grouped_dataset,
    compute_phash,
    compute_sha256,
    discover_images,
    scan_images,
)


def _write_pattern(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="JPEG", quality=92)


def _source_tree(root: Path) -> Path:
    source = root / "source"
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = source / class_name
        class_dir.mkdir(parents=True)
        for sample_index in range(6):
            _write_pattern(
                class_dir / f"{class_name}_{sample_index}.jpg",
                class_index * 100 + sample_index,
            )
    shutil.copy2(
        source / "Cyst" / "Cyst_0.jpg",
        source / "Cyst" / "Cyst_exact_copy.jpg",
    )
    return source


def test_sha256_detects_exact_duplicates(tmp_path):
    original = tmp_path / "original.jpg"
    copied = tmp_path / "copy.jpg"
    original.write_bytes(b"same image bytes")
    shutil.copy2(original, copied)
    assert compute_sha256(original) == compute_sha256(copied)


def test_phash_components_join_neighbors_transitively_and_exact_duplicates():
    records = [
        ImageRecord(Path("a.jpg"), "a.jpg", "Cyst", "sha-a", 0b000000),
        ImageRecord(Path("b.jpg"), "b.jpg", "Cyst", "sha-b", 0b000011),
        ImageRecord(Path("c.jpg"), "c.jpg", "Cyst", "sha-c", 0b000111),
        ImageRecord(Path("d.jpg"), "d.jpg", "Normal", "sha-d", 0xFFFFFFFFFFFFFFFF),
        ImageRecord(Path("e.jpg"), "e.jpg", "Cyst", "sha-a", 0b000000),
    ]
    components, edges = build_components(records, distance_threshold=2)
    assert components.find(0) == components.find(1) == components.find(2)
    assert components.find(0) == components.find(4)
    assert components.find(3) != components.find(0)
    assert len(edges) == 2


def test_class_coverage_group_isolation_and_targeted_proportions():
    records = []
    groups = []
    for class_name in CLASS_NAMES:
        for index in range(20):
            group = f"{class_name}_{index}"
            groups.append(group)
            records.append(
                ImageRecord(Path(f"{group}.jpg"), f"{group}.jpg", class_name, group, index)
            )
    assignment = assign_group_splits(records, groups)
    assert set(assignment) == set(groups)
    assert set(assignment.values()) == {"train", "validation", "test"}
    for class_name in CLASS_NAMES:
        assigned = [
            assignment[group]
            for group in groups
            if group.startswith(f"{class_name}_")
        ]
        assert set(assigned) == {"train", "validation", "test"}
        assert 0.60 <= assigned.count("train") / len(assigned) <= 0.80
        assert 0.05 <= assigned.count("validation") / len(assigned) <= 0.25
        assert 0.05 <= assigned.count("test") / len(assigned) <= 0.25


def test_missing_dataset_directory_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_images(tmp_path / "missing")


def test_missing_class_directory_is_rejected(tmp_path):
    source = tmp_path / "source"
    for class_name in CLASS_NAMES[:-1]:
        (source / class_name).mkdir(parents=True)
    with pytest.raises(ValueError, match="exactly class directories"):
        discover_images(source)


def test_missing_image_file_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for class_name in CLASS_NAMES:
        (source / class_name).mkdir(parents=True)
    image = source / "Cyst" / "image.jpg"
    _write_pattern(image, 1)

    def remove_before_hash(path):
        path.unlink()
        raise FileNotFoundError(path)

    monkeypatch.setattr("dl.preprocessing.grouping.compute_sha256", remove_before_hash)
    with pytest.raises(ValueError, match="Could not read or hash image"):
        scan_images(source)


def test_phash_is_stable_for_same_image(tmp_path):
    image = tmp_path / "image.jpg"
    _write_pattern(image, 42)
    assert compute_phash(image) == compute_phash(image)


def test_complete_pipeline_is_deterministic_and_preserves_source(tmp_path):
    source = _source_tree(tmp_path)
    source_file = source / "Cyst" / "Cyst_0.jpg"
    source_hash = compute_sha256(source_file)
    first = build_grouped_dataset(
        source, tmp_path / "first_grouped", tmp_path / "first_reports"
    )
    second = build_grouped_dataset(
        source, tmp_path / "second_grouped", tmp_path / "second_reports"
    )
    assert first["split_counts"] == second["split_counts"]
    assert first["class_counts"] == second["class_counts"]
    assert first["cluster_count"] == second["cluster_count"]
    first_manifest = pd.read_csv(
        tmp_path / "first_reports" / "grouped_split_manifest.csv"
    )
    second_manifest = pd.read_csv(
        tmp_path / "second_reports" / "grouped_split_manifest.csv"
    )
    columns = [
        "filename",
        "class",
        "destination_split",
        "near_duplicate_group_id",
    ]
    assert first_manifest[columns].equals(second_manifest[columns])
    assert compute_sha256(source_file) == source_hash
    assert first["exact_duplicate_pair_count"] == 1
    audit = pd.read_csv(
        tmp_path / "first_reports" / "duplicate_analysis.csv"
    )
    exact = audit[audit["is_exact_duplicate"]]
    assert len(exact) == 2
    assert exact["near_duplicate_group_id"].nunique() == 1
    assert audit.groupby("near_duplicate_group_id")["class"].nunique().max() >= 1
    for path in (tmp_path / "first_grouped").rglob("*.jpg"):
        assert path.is_file()


def test_existing_output_is_not_overwritten(tmp_path):
    source = _source_tree(tmp_path)
    processed = tmp_path / "grouped"
    processed.mkdir()
    marker = processed / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_grouped_dataset(source, processed, tmp_path / "reports")
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_output_trees_must_be_separate_from_source_and_each_other(tmp_path):
    source = _source_tree(tmp_path)
    source_images_before = sorted(path.relative_to(source) for path in source.rglob("*"))

    with pytest.raises(ValueError, match="Reports directory must be separate"):
        build_grouped_dataset(source, tmp_path / "processed", source / "reports")
    with pytest.raises(ValueError, match="Reports directory must be separate"):
        build_grouped_dataset(source, tmp_path / "processed", tmp_path)
    with pytest.raises(ValueError, match="must use separate output trees"):
        build_grouped_dataset(
            source, tmp_path / "outputs", tmp_path / "outputs" / "reports"
        )

    assert (
        sorted(path.relative_to(source) for path in source.rglob("*"))
        == source_images_before
    )
