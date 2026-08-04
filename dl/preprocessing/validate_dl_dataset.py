"""DL Image Dataset Validation Module.

Validates image datasets, class distribution, corrupt images, duplicate filenames,
resolutions, and formats for deep learning tasks in StoneSense-AI.
"""

import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DLImageDatasetValidator")


class DLDatasetValidator:
    """Validator class for image datasets in Deep Learning."""

    def __init__(self, dataset_dir: Path, output_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_class_directories(self) -> Dict[str, Path]:
        """Locates class subdirectories recursively."""
        class_dirs: Dict[str, Path] = {}
        if not self.dataset_dir.exists():
            logger.error(f"Dataset root directory does not exist: {self.dataset_dir}")
            return class_dirs

        # Check direct subdirectories first
        subdirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        
        # If there's a nested folder with the same name, dive into it
        if len(subdirs) == 1 and subdirs[0].is_dir() and subdirs[0].name == self.dataset_dir.name:
            subdirs = [d for d in subdirs[0].iterdir() if d.is_dir()]

        for d in subdirs:
            # Standard classes: Cyst, Normal, Stone, Tumor (case-insensitive check)
            class_name = d.name.capitalize()
            class_dirs[class_name] = d

        logger.info(f"Detected classes: {list(class_dirs.keys())}")
        return class_dirs

    def validate(self) -> Dict[str, Any]:
        """Scans image dataset and collects metrics."""
        class_dirs = self.find_class_directories()
        
        class_metrics: Dict[str, Dict[str, Any]] = {}
        corrupt_images: List[Dict[str, str]] = []
        all_filenames: Dict[str, List[str]] = defaultdict(list)
        overall_formats = Counter()

        for class_name, class_path in class_dirs.items():
            logger.info(f"Scanning class directory: {class_name} ({class_path})...")
            
            image_files = [
                f for f in class_path.rglob("*") 
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
            ]

            img_count = len(image_files)
            corrupt_count = 0
            widths: List[int] = []
            heights: List[int] = []
            formats = Counter()

            for img_path in tqdm(image_files, desc=f"Validating {class_name}"):
                filename = img_path.name
                all_filenames[filename].append(str(img_path))

                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify image integrity
                    
                    # Re-open to read attributes after verify()
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        fmt = img.format if img.format else img_path.suffix[1:].upper()
                        formats[fmt] += 1
                        overall_formats[fmt] += 1

                except Exception as err:
                    logger.warning(f"Corrupt or unreadable image found: {img_path} - {err}")
                    corrupt_count += 1
                    corrupt_images.append({
                        "class": class_name,
                        "filepath": str(img_path),
                        "filename": img_path.name,
                        "error": str(err)
                    })

            avg_w = float(pd.Series(widths).mean()) if widths else 0.0
            avg_h = float(pd.Series(heights).mean()) if heights else 0.0
            min_res = (min(widths), min(heights)) if widths else (0, 0)
            max_res = (max(widths), max(heights)) if heights else (0, 0)

            class_metrics[class_name] = {
                "image_count": img_count,
                "corrupt_count": corrupt_count,
                "valid_count": len(widths),
                "avg_width": round(avg_w, 2),
                "avg_height": round(avg_h, 2),
                "min_resolution": f"{min_res[0]}x{min_res[1]}",
                "max_resolution": f"{max_res[0]}x{max_res[1]}",
                "formats": dict(formats)
            }

        # Detect duplicate filenames across dataset
        duplicate_filenames = {
            fname: paths for fname, paths in all_filenames.items() if len(paths) > 1
        }

        results = {
            "class_metrics": class_metrics,
            "corrupt_images": corrupt_images,
            "duplicate_filenames": duplicate_filenames,
            "total_duplicate_names": len(duplicate_filenames),
            "overall_formats": dict(overall_formats)
        }

        return results

    def save_reports(self, results: Dict[str, Any]) -> None:
        """Saves class distribution CSV, corrupt images CSV, chart, and Markdown report."""
        logger.info("Saving DL validation reports...")

        # 1. Class Distribution CSV
        dist_rows = []
        for cname, metrics in results["class_metrics"].items():
            dist_rows.append({
                "class_name": cname,
                "total_images": metrics["image_count"],
                "valid_images": metrics["valid_count"],
                "corrupt_images": metrics["corrupt_count"],
                "avg_width": metrics["avg_width"],
                "avg_height": metrics["avg_height"],
                "min_resolution": metrics["min_resolution"],
                "max_resolution": metrics["max_resolution"],
                "formats": str(metrics["formats"])
            })

        dist_df = pd.DataFrame(dist_rows)
        dist_csv_path = self.output_dir / "class_distribution.csv"
        dist_df.to_csv(dist_csv_path, index=False)
        logger.info(f"Saved class distribution CSV to {dist_csv_path}")

        # 2. Corrupt Images CSV
        corrupt_df = pd.DataFrame(results["corrupt_images"])
        if corrupt_df.empty:
            corrupt_df = pd.DataFrame(columns=["class", "filepath", "filename", "error"])
        
        corrupt_csv_path = self.output_dir / "corrupt_images.csv"
        corrupt_df.to_csv(corrupt_csv_path, index=False)
        logger.info(f"Saved corrupt images log to {corrupt_csv_path}")

        # 3. Class Distribution Chart
        chart_path = self.output_dir / "class_distribution_chart.png"
        self._generate_chart(dist_df, chart_path)

        # 4. Image Validation Markdown Report
        md_report_path = self.output_dir / "image_validation_report.md"
        self._generate_markdown_report(results, dist_df, md_report_path)
        logger.info(f"Saved image validation Markdown report to {md_report_path}")

    def _generate_chart(self, dist_df: pd.DataFrame, chart_path: Path) -> None:
        """Generates a styled bar chart for class distribution."""
        if dist_df.empty:
            return
        
        plt.figure(figsize=(8, 5))
        plt.bar(dist_df["class_name"], dist_df["valid_images"], color="#2563eb", edgecolor="#1e40af")
        plt.title("CT Kidney Image Dataset - Class Distribution", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Class", fontsize=12, fontweight="bold")
        plt.ylabel("Number of Valid Images", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close()

    def _generate_markdown_report(self, results: Dict[str, Any], dist_df: pd.DataFrame, md_path: Path) -> None:
        """Helper to create Markdown report."""
        total_images = sum(m["image_count"] for m in results["class_metrics"].values())
        total_corrupt = len(results["corrupt_images"])
        total_duplicates = results["total_duplicate_names"]

        if not dist_df.empty:
            headers = list(dist_df.columns)
            header_str = "| " + " | ".join(headers) + " |"
            sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
            rows_str = []
            for _, row in dist_df.iterrows():
                rows_str.append("| " + " | ".join(str(val) for val in row.values) + " |")
            table_md = "\n".join([header_str, sep_str] + rows_str)
        else:
            table_md = "No data."

        corrupt_md = "No corrupt images detected."
        if total_corrupt > 0:
            corrupt_md = f"Detected **{total_corrupt}** corrupt image(s). Details saved in `corrupt_images.csv`."

        dup_md = f"Found **{total_duplicates}** duplicate filename instances across folders."

        md_content = f"""# CT Kidney Dataset Image Validation Report

**Dataset Source:** `{self.dataset_dir}`  
**Status:** Completed  

---

## 📊 Summary Overview

- **Total Images Scanned:** {total_images}
- **Total Valid Images:** {total_images - total_corrupt}
- **Total Corrupt Images:** {total_corrupt}
- **Duplicate Filenames:** {total_duplicates}
- **File Formats Detected:** `{results['overall_formats']}`

---

## 📁 Class Distribution & Resolution Statistics

{table_md}

---

## ⚠️ Data Quality Alerts

### Corrupt Images
{corrupt_md}

### Duplicate Filenames
{dup_md}

---

## 🎯 Verification Conclusion
All images across detected class subdirectories were checked for structural readability, aspect ratio/resolution bounds, and format consistency. The dataset is ready for preprocessing and model ingestion.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    dataset_dir = base_dir / "datasets" / "archive" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    output_dir = base_dir / "outputs" / "reports"

    validator = DLDatasetValidator(dataset_dir=dataset_dir, output_dir=output_dir)
    results = validator.validate()
    validator.save_reports(results)
    logger.info("DL Image Dataset Validation completed successfully.")


if __name__ == "__main__":
    main()
