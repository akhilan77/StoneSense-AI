"""ML Dataset Validation Module.

Validates structure, data quality, statistics, missing values, duplicates,
and correlations for machine learning datasets in StoneSense-AI.
"""

from pathlib import Path
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MLDatasetValidator")


class MLDatasetValidator:
    """Validator class for tabular machine learning datasets."""

    def __init__(self, data_path: Path, output_dir: Path):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from CSV file."""
        if not self.data_path.exists():
            logger.error(f"Dataset path does not exist: {self.data_path}")
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        logger.info(f"Loading ML dataset from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset successfully loaded with shape: {self.df.shape}")
        return self.df

    def validate(self) -> Dict[str, Any]:
        """Performs complete validation on the dataset."""
        if self.df.empty:
            self.load_data()

        logger.info("Performing structural & data quality checks...")
        
        # Basic Structural Stats
        num_rows, num_cols = self.df.shape
        column_names = list(self.df.columns)
        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        
        # Missing values
        missing_series = self.df.isnull().sum()
        missing_pct = (missing_series / num_rows) * 100
        missing_df = pd.DataFrame({
            'column': missing_series.index,
            'missing_count': missing_series.values,
            'missing_percentage': missing_pct.values
        })
        
        # Duplicate rows
        duplicate_rows = int(self.df.duplicated().sum())

        # Invalid numeric values (NaN, Inf, -Inf)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        invalid_numeric: Dict[str, Dict[str, int]] = {}
        for col in numeric_cols:
            col_data = self.df[col]
            n_nan = int(col_data.isnull().sum())
            n_inf = int(np.isinf(col_data).sum()) if np.issubdtype(col_data.dtype, np.floating) else 0
            invalid_numeric[col] = {"nan_count": n_nan, "inf_count": n_inf}

        # Descriptive Statistics
        desc_stats = self.df.describe(include='all').T.reset_index().rename(columns={'index': 'column'})

        # Target class distribution (if 'target' or 'Class' exists)
        target_col = None
        for candidate in ['Class', 'target', 'label', 'diag']:
            if candidate in self.df.columns:
                target_col = candidate
                break
        
        target_dist: Dict[str, int] = {}
        if target_col:
            target_dist = self.df[target_col].value_counts().to_dict()

        # Unique values for categorical columns
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        unique_categorical: Dict[str, int] = {col: int(self.df[col].nunique()) for col in cat_cols}

        # Correlation Matrix
        corr_matrix = None
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()

        results = {
            "dataset_shape": (num_rows, num_cols),
            "column_names": column_names,
            "data_types": dtypes,
            "missing_df": missing_df,
            "duplicate_rows": duplicate_rows,
            "invalid_numeric": invalid_numeric,
            "descriptive_stats": desc_stats,
            "target_column": target_col,
            "target_distribution": target_dist,
            "unique_categorical": unique_categorical,
            "correlation_matrix": corr_matrix
        }

        return results

    def save_reports(self, results: Dict[str, Any]) -> None:
        """Saves summary CSV, missing values CSV, and HTML validation report."""
        logger.info("Saving validation outputs and reports...")

        # 1. Save missing_values.csv
        missing_csv_path = self.output_dir / "missing_values.csv"
        results["missing_df"].to_csv(missing_csv_path, index=False)
        logger.info(f"Saved missing values report to {missing_csv_path}")

        # 2. Save validation_summary.csv
        shape = results["dataset_shape"]
        summary_data = [
            {"metric": "Total Rows", "value": shape[0]},
            {"metric": "Total Columns", "value": shape[1]},
            {"metric": "Duplicate Rows", "value": results["duplicate_rows"]},
            {"metric": "Target Column", "value": str(results["target_column"])},
        ]
        for col, count in results["target_distribution"].items():
            summary_data.append({"metric": f"Target Class Count ({col})", "value": count})

        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / "validation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved validation summary CSV to {summary_csv_path}")

        # 3. Generate and save validation_report.html
        html_report_path = self.output_dir / "validation_report.html"
        self._generate_html_report(results, html_report_path)
        logger.info(f"Saved HTML validation report to {html_report_path}")

    def _generate_html_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Helper to create a formatted HTML report."""
        shape = results["dataset_shape"]
        desc_html = results["descriptive_stats"].to_html(classes="table table-striped table-bordered", index=False)
        missing_html = results["missing_df"].to_html(classes="table table-striped table-bordered", index=False)
        
        corr_html = ""
        if results["correlation_matrix"] is not None:
            corr_html = results["correlation_matrix"].to_html(classes="table table-striped table-bordered")

        target_html = "<ul>"
        for k, v in results["target_distribution"].items():
            target_html += f"<li><strong>{k}</strong>: {v}</li>"
        target_html += "</ul>"

        cat_html = "<ul>"
        for k, v in results["unique_categorical"].items():
            cat_html += f"<li><strong>{k}</strong>: {v} unique values</li>"
        cat_html += "</ul>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Dataset Validation Report - StoneSense-AI</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {{ background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; }}
        .card {{ margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; }}
        .card-header {{ background-color: #1e293b; color: #ffffff; font-weight: 600; font-size: 1.1rem; }}
        table {{ font-size: 0.9rem; }}
        h1 {{ color: #0f172a; margin-bottom: 25px; font-weight: 700; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1>StoneSense-AI: ML Dataset Validation Report</h1>
        <p class="text-muted">Dataset Source: <code>{self.data_path}</code></p>
        
        <!-- Summary Cards -->
        <div class="row">
            <div class="col-md-3">
                <div class="card text-center p-3">
                    <h3>{shape[0]}</h3>
                    <span class="text-muted">Total Rows</span>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center p-3">
                    <h3>{shape[1]}</h3>
                    <span class="text-muted">Total Columns</span>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center p-3">
                    <h3>{results["duplicate_rows"]}</h3>
                    <span class="text-muted">Duplicate Rows</span>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center p-3">
                    <h3>{len(results["target_distribution"])}</h3>
                    <span class="text-muted">Target Classes</span>
                </div>
            </div>
        </div>

        <!-- Target & Categorical Breakdown -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Target Distribution ({results['target_column']})</div>
                    <div class="card-body">{target_html}</div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Categorical Unique Values</div>
                    <div class="card-body">{cat_html}</div>
                </div>
            </div>
        </div>

        <!-- Missing Values -->
        <div class="card">
            <div class="card-header">Missing Values Analysis</div>
            <div class="card-body table-responsive">{missing_html}</div>
        </div>

        <!-- Descriptive Statistics -->
        <div class="card">
            <div class="card-header">Descriptive Statistics</div>
            <div class="card-body table-responsive">{desc_html}</div>
        </div>

        <!-- Correlation Matrix -->
        {f'<div class="card"><div class="card-header">Correlation Matrix</div><div class="card-body table-responsive">{corr_html}</div></div>' if corr_html else ''}

    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_file = base_dir / "datasets" / "kidneyData.csv"
    output_dir = base_dir / "outputs" / "reports"
    
    validator = MLDatasetValidator(data_path=data_file, output_dir=output_dir)
    results = validator.validate()
    validator.save_reports(results)
    logger.info("ML Dataset Validation completed successfully.")


if __name__ == "__main__":
    main()
