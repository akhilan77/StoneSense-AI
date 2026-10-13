"""Generate duplicate-aware CT split data and its audit reports."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

if __package__:
    from .grouping import build_grouped_dataset
else:
    from grouping import build_grouped_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Group SHA-256 exact duplicates and pHash-near-duplicates, then "
            "assign whole connected components to 70/15/15 splits."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            r"D:\Data\StoneSenseAI\datasets\ct-kidney-kaggle\extracted"
            r"\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
            r"\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        ),
        help="Directory containing the Cyst, Normal, Stone, and Tumor folders.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=project_root / "dl" / "processed_grouped",
        help="New grouped image tree. Existing destinations are never overwritten.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=project_root / "dl" / "outputs" / "reports",
    )
    parser.add_argument("--phash-distance", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = build_grouped_dataset(
        source_dir=args.source_dir,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        distance_threshold=args.phash_distance,
    )
    print(f"Images: {report['image_count']}")
    print(f"Near-duplicate groups: {report['cluster_count']}")
    print(f"Exact duplicate pairs: {report['exact_duplicate_pair_count']}")
    print(f"Near-duplicate pHash edges: {report['near_duplicate_phash_edge_count']}")
    print(f"Split counts: {report['split_counts']}")
    print(f"Class counts: {report['class_counts']}")
    print(f"Manifest: {args.reports_dir / 'grouped_split_manifest.csv'}")
    print(f"Duplicate audit: {args.reports_dir / 'duplicate_analysis.csv'}")
    print(f"pHash clusters: {args.reports_dir / 'phash_clusters.json'}")
    print(f"pHash edges: {args.reports_dir / 'phash_duplicates.csv'}")
    print(f"Grouped images: {args.processed_dir}")


if __name__ == "__main__":
    main()
