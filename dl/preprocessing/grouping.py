"""Exact and perceptual image grouping for leakage-aware CT dataset splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

LOGGER = logging.getLogger(__name__)

CLASS_NAMES = ("Cyst", "Normal", "Stone", "Tumor")
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PHASH_SIZE = 8
PHASH_RESIZE = 32


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    filename: str
    class_name: str
    sha256: str
    phash: int


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_phash(path: Path, hash_size: int = PHASH_SIZE) -> int:
    """Compute a 64-bit DCT perceptual hash (same median-bit convention as ImageHash)."""
    with Image.open(path) as source:
        pixels = np.asarray(
            source.convert("L").resize(
                (PHASH_RESIZE, PHASH_RESIZE), Image.Resampling.LANCZOS
            ),
            dtype=np.float64,
        )
    coefficients = dctn(pixels, type=2, norm="ortho")[:hash_size, :hash_size]
    low_frequencies = coefficients.ravel()
    median = float(np.median(low_frequencies))
    bits = low_frequencies > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def discover_images(source_dir: Path) -> list[tuple[Path, str]]:
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset source directory does not exist: {source_dir}")

    # Kaggle releases wrap class folders in a same-named directory.
    child_dirs = [path for path in source_dir.iterdir() if path.is_dir()]
    if len(child_dirs) == 1 and child_dirs[0].name == source_dir.name:
        source_dir = child_dirs[0]

    class_dirs = {path.name: path for path in source_dir.iterdir() if path.is_dir()}
    if set(class_dirs) != set(CLASS_NAMES):
        raise ValueError(
            f"Expected exactly class directories {list(CLASS_NAMES)} in {source_dir}; "
            f"found {sorted(class_dirs)}"
        )

    records: list[tuple[Path, str]] = []
    for class_name in CLASS_NAMES:
        for path in sorted(class_dirs[class_name].rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((path.resolve(), class_name))
    if not records:
        raise ValueError(f"No supported image files found below {source_dir}")

    duplicate_filenames = pd.Series([path.name for path, _ in records])
    if duplicate_filenames.duplicated().any():
        duplicates = duplicate_filenames[duplicate_filenames.duplicated()].unique().tolist()
        raise ValueError(
            "Image basenames must be globally unique for the downstream manifest "
            f"contract; collisions include {duplicates[:10]}"
        )
    return records


def scan_images(source_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path, class_name in discover_images(source_dir):
        try:
            with Image.open(path) as image:
                image.verify()
            sha256 = compute_sha256(path)
            phash = compute_phash(path)
        except Exception as exc:
            raise ValueError(f"Could not read or hash image {path}: {exc}") from exc
        records.append(ImageRecord(path, path.name, class_name, sha256, phash))
    return records


def _hamming_masks(hash_bits: int) -> list[int]:
    one_bit = [1 << bit for bit in range(hash_bits)]
    two_bit = [
        (1 << left) | (1 << right)
        for left in range(hash_bits)
        for right in range(left + 1, hash_bits)
    ]
    return one_bit + two_bit


def build_components(
    records: list[ImageRecord], distance_threshold: int = 2
) -> tuple[UnionFind, list[tuple[int, int, int]]]:
    """Join exact duplicates and pHash-neighbor edges into connected components."""
    if distance_threshold not in (0, 1, 2):
        raise ValueError("Supported pHash distance thresholds are 0, 1, or 2.")

    components = UnionFind(len(records))
    by_sha256: dict[str, list[int]] = defaultdict(list)
    by_phash: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_sha256[record.sha256].append(index)
        by_phash[record.phash].append(index)

    for duplicate_indices in by_sha256.values():
        for index in duplicate_indices[1:]:
            components.union(duplicate_indices[0], index)

    # A zero-distance pHash collision also constitutes a graph edge.
    for matching_indices in by_phash.values():
        for index in matching_indices[1:]:
            components.union(matching_indices[0], index)

    hashes = sorted(by_phash)
    hash_to_representative = {value: by_phash[value][0] for value in hashes}
    masks = _hamming_masks(PHASH_SIZE * PHASH_SIZE)
    masks = [mask for mask in masks if mask.bit_count() <= distance_threshold]
    edges: list[tuple[int, int, int]] = []
    for left_hash in hashes:
        for mask in masks:
            right_hash = left_hash ^ mask
            if right_hash <= left_hash or right_hash not in hash_to_representative:
                continue
            edges.append((left_hash, right_hash, mask.bit_count()))
            components.union(
                hash_to_representative[left_hash],
                hash_to_representative[right_hash],
            )
    return components, edges


def _stable_cluster_ids(records: list[ImageRecord], components: UnionFind) -> list[str]:
    member_hashes: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        member_hashes[components.find(index)].append(record.sha256)
    root_to_cluster = {
        root: f"phash_{min(hashes)}" for root, hashes in member_hashes.items()
    }
    return [root_to_cluster[components.find(index)] for index in range(len(records))]


def assign_group_splits(
    records: list[ImageRecord],
    cluster_ids: list[str],
    proportions: dict[str, float] | None = None,
) -> dict[str, str]:
    """Allocate complete pHash components with deterministic class-stratified balancing."""
    ratios = proportions or SPLIT_PROPORTIONS
    if (
        set(ratios) != set(SPLIT_NAMES)
        or any(value <= 0 for value in ratios.values())
        or not np.isclose(sum(ratios.values()), 1.0)
    ):
        raise ValueError(f"Split proportions must define {SPLIT_NAMES} and sum to one.")

    cluster_classes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    class_totals = {class_name: 0 for class_name in CLASS_NAMES}
    for record, cluster_id in zip(records, cluster_ids):
        cluster_classes[cluster_id][record.class_name] += 1
        class_totals[record.class_name] += 1

    for class_name in CLASS_NAMES:
        class_groups = [
            cluster_id
            for cluster_id, counts in cluster_classes.items()
            if counts.get(class_name, 0)
        ]
        if len(class_groups) < len(SPLIT_NAMES):
            raise ValueError(
                f"Cannot preserve {class_name} coverage in all splits: only "
                f"{len(class_groups)} independent duplicate groups."
            )

    assignment: dict[str, str] = {}
    current = {
        split: {class_name: 0 for class_name in CLASS_NAMES}
        for split in SPLIT_NAMES
    }
    target = {
        split: {
            class_name: class_totals[class_name] * ratios[split]
            for class_name in CLASS_NAMES
        }
        for split in SPLIT_NAMES
    }

    # Reserve small deterministic groups first to guarantee every class is represented.
    for split in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            if current[split][class_name]:
                continue
            available = [
                cluster_id
                for cluster_id, counts in cluster_classes.items()
                if cluster_id not in assignment and counts.get(class_name, 0)
            ]
            if not available:
                raise ValueError(f"Unable to reserve {class_name} coverage for {split}.")
            chosen = min(
                available,
                key=lambda cluster_id: (
                    sum(cluster_classes[cluster_id].values()),
                    cluster_id,
                ),
            )
            assignment[chosen] = split
            for label, count in cluster_classes[chosen].items():
                current[split][label] += count

    def objective(split: str, group: dict[str, int]) -> float:
        score = 0.0
        for candidate_split in SPLIT_NAMES:
            for class_name in CLASS_NAMES:
                count = current[candidate_split][class_name]
                if candidate_split == split:
                    count += group.get(class_name, 0)
                desired = target[candidate_split][class_name]
                score += ((count - desired) / max(desired, 1.0)) ** 2
        return score

    pending = sorted(
        (cluster_id for cluster_id in cluster_classes if cluster_id not in assignment),
        key=lambda cluster_id: (
            -sum(cluster_classes[cluster_id].values()),
            cluster_id,
        ),
    )
    for cluster_id in pending:
        group = cluster_classes[cluster_id]
        chosen = min(
            SPLIT_NAMES,
            key=lambda split: (objective(split, group), SPLIT_NAMES.index(split)),
        )
        assignment[cluster_id] = chosen
        for class_name, count in group.items():
            current[chosen][class_name] += count
    return assignment


def build_grouped_dataset(
    source_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    distance_threshold: int = 2,
    proportions: dict[str, float] | None = None,
) -> dict[str, object]:
    """Hash source images, assign groups, then write reports and a copied split tree."""
    source_dir = Path(source_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    reports_dir = Path(reports_dir).resolve()
    report_paths = {
        reports_dir / "grouped_split_manifest.csv",
        reports_dir / "duplicate_analysis.csv",
        reports_dir / "phash_clusters.json",
        reports_dir / "phash_duplicates.csv",
    }
    existing = [path for path in [processed_dir, *report_paths] if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing grouped outputs: "
            + ", ".join(str(path) for path in existing)
        )
    if processed_dir == source_dir or source_dir in processed_dir.parents:
        raise ValueError("Processed output must not be inside the source image tree.")
    if processed_dir in source_dir.parents:
        raise ValueError("Processed output must not contain the source image tree.")
    if (
        reports_dir == source_dir
        or source_dir in reports_dir.parents
        or reports_dir in source_dir.parents
    ):
        raise ValueError("Reports directory must be separate from the source image tree.")
    if (
        reports_dir == processed_dir
        or reports_dir in processed_dir.parents
        or processed_dir in reports_dir.parents
    ):
        raise ValueError("Processed images and reports must use separate output trees.")

    records = scan_images(source_dir)
    components, near_edges = build_components(records, distance_threshold)
    cluster_ids = _stable_cluster_ids(records, components)
    cluster_assignment = assign_group_splits(records, cluster_ids, proportions)
    split_names = [cluster_assignment[cluster_id] for cluster_id in cluster_ids]

    sha_counts: dict[str, int] = defaultdict(int)
    for record in records:
        sha_counts[record.sha256] += 1
    duplicate_audit = pd.DataFrame(
        {
            "filename": [record.filename for record in records],
            "source_path": [str(record.path) for record in records],
            "class": [record.class_name for record in records],
            "sha256": [record.sha256 for record in records],
            "is_exact_duplicate": [
                sha_counts[record.sha256] > 1 for record in records
            ],
            "exact_duplicate_group_id": [
                f"sha256_{record.sha256}" for record in records
            ],
            "phash": [f"{record.phash:016x}" for record in records],
            "near_duplicate_group_id": cluster_ids,
        }
    )
    manifest = duplicate_audit[
        ["filename", "source_path", "class", "near_duplicate_group_id"]
    ].copy()
    manifest["destination_split"] = split_names
    manifest["destination_path"] = [
        str(processed_dir / split / class_name / filename)
        for split, class_name, filename in zip(
            split_names, manifest["class"], manifest["filename"]
        )
    ]
    manifest = manifest[
        [
            "filename",
            "source_path",
            "class",
            "destination_split",
            "near_duplicate_group_id",
            "destination_path",
        ]
    ]

    # Each connected component is indivisible, including edges chained by transitivity.
    cluster_rows: dict[str, dict[str, object]] = {}
    for cluster_id, group in duplicate_audit.groupby("near_duplicate_group_id", sort=True):
        phashes = group["phash"].tolist()
        cluster_rows[cluster_id] = {
            "near_duplicate_group_id": cluster_id,
            "destination_split": cluster_assignment[cluster_id],
            "image_count": int(len(group)),
            "class_counts": group["class"].value_counts().sort_index().to_dict(),
            "images": group[
                ["filename", "class", "sha256", "phash"]
            ].to_dict(orient="records"),
        }

    class_counts = (
        manifest.groupby(["destination_split", "class"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SPLIT_NAMES, columns=CLASS_NAMES, fill_value=0)
    )
    split_counts = class_counts.sum(axis=1).to_dict()
    for split in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            if class_counts.loc[split, class_name] <= 0:
                raise ValueError(f"Class coverage missing: {split}/{class_name}")

    manifest["source_exists"] = manifest["source_path"].map(
        lambda path: Path(path).is_file()
    )
    if not manifest["source_exists"].all():
        raise FileNotFoundError("One or more source images disappeared during audit.")
    manifest.drop(columns=["source_exists"], inplace=True)

    # Preflight individual destination filenames before creating any outputs.
    for row in manifest.itertuples(index=False):
        destination = Path(row.destination_path)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite image: {destination}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest.itertuples(index=False):
        destination = Path(row.destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.source_path, destination)

    manifest_path = reports_dir / "grouped_split_manifest.csv"
    duplicate_path = reports_dir / "duplicate_analysis.csv"
    clusters_path = reports_dir / "phash_clusters.json"
    edges_path = reports_dir / "phash_duplicates.csv"
    manifest.to_csv(manifest_path, index=False)
    duplicate_audit.to_csv(duplicate_path, index=False)
    phash_to_record = {}
    phash_to_cluster = {}
    for index, record in enumerate(records):
        phash_to_record.setdefault(record.phash, record)
        phash_to_cluster.setdefault(record.phash, cluster_ids[index])
    pd.DataFrame(
        [
            {
                "left_filename": phash_to_record[left].filename,
                "left_sha256": phash_to_record[left].sha256,
                "left_phash": f"{left:016x}",
                "right_filename": phash_to_record[right].filename,
                "right_sha256": phash_to_record[right].sha256,
                "right_phash": f"{right:016x}",
                "hamming_distance": distance,
                "near_duplicate_group_id": phash_to_cluster[left],
            }
            for left, right, distance in near_edges
        ],
        columns=[
            "left_filename",
            "left_sha256",
            "left_phash",
            "right_filename",
            "right_sha256",
            "right_phash",
            "hamming_distance",
            "near_duplicate_group_id",
        ],
    ).to_csv(edges_path, index=False)

    report = {
        "algorithm": "64-bit DCT pHash, connected components",
        "phash_hamming_distance_threshold": distance_threshold,
        "exact_duplicate_hash": "SHA-256",
        "split_proportions_target": proportions or SPLIT_PROPORTIONS,
        "split_counts": {split: int(split_counts[split]) for split in SPLIT_NAMES},
        "class_counts": {
            split: {
                class_name: int(class_counts.loc[split, class_name])
                for class_name in CLASS_NAMES
            }
            for split in SPLIT_NAMES
        },
        "image_count": len(records),
        "cluster_count": len(cluster_rows),
        "exact_duplicate_image_count": int(
            sum(count for count in sha_counts.values() if count > 1)
        ),
        "exact_duplicate_pair_count": int(
            sum(count * (count - 1) // 2 for count in sha_counts.values())
        ),
        "near_duplicate_phash_edge_count": len(near_edges),
        "groups": list(cluster_rows.values()),
        "limitations": [
            "Near-duplicate groups use image-level pHash similarity, not patient identifiers.",
            "Patient-level isolation cannot be established because this dataset has no patient IDs.",
            "Connected components are transitive; endpoints may differ by more than the threshold.",
        ],
    }
    clusters_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    LOGGER.info("Grouped split counts: %s", report["split_counts"])
    LOGGER.info("Class counts: %s", report["class_counts"])
    LOGGER.info(
        "Images=%d groups=%d exact_duplicate_pairs=%d near_phash_edges=%d",
        len(records),
        len(cluster_rows),
        report["exact_duplicate_pair_count"],
        len(near_edges),
    )
    return report
