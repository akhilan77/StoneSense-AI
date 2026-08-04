"""Main Machine Learning Preprocessing Pipeline Execution Script.

Cleans dataset, fits transformers, splits dataset (70/15/15), saves artifacts,
and exports processed CSV files.
"""

from pathlib import Path
import json
import logging
import pandas as pd

from tabular_preprocessor import TabularPreprocessor
from feature_engineering import extract_metadata
from split_dataset import split_tabular_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MLPreprocessingPipeline")


def run_ml_preprocessing_pipeline() -> None:
    """Executes end-to-end ML preprocessing pipeline."""
    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "datasets" / "kidneyData.csv"
    artifacts_dir = base_dir / "artifacts"
    processed_dir = base_dir / "processed"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 3 ML Preprocessing Pipeline...")

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        raise FileNotFoundError(f"File not found: {dataset_path}")

    # 1. Load Data
    raw_df = pd.read_csv(dataset_path)

    # 2. Instantiate and clean data
    preprocessor = TabularPreprocessor()
    num_cols, cat_cols, target_col = preprocessor.auto_detect_columns(raw_df)
    cleaned_df = preprocessor.clean_data(raw_df)

    # 3. Fit ColumnTransformer pipeline
    X_features = cleaned_df.drop(columns=[target_col])
    preprocessor.fit_transform(X_features)

    # 4. Save ML Preprocessing artifacts
    preprocessor.save_artifacts(artifacts_dir)

    # 5. Split Dataset (70% Train, 15% Val, 15% Test)
    train_df, val_df, test_df = split_tabular_dataset(cleaned_df, target_column=target_col, random_state=42)

    # 6. Export Processed Datasets
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "validation.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Exported train dataset to {train_path}")
    logger.info(f"Exported validation dataset to {val_path}")
    logger.info(f"Exported test dataset to {test_path}")

    # 7. Generate and save preprocessing metadata JSON
    metadata = extract_metadata(
        df=cleaned_df,
        target_col=target_col,
        num_cols=num_cols,
        cat_cols=cat_cols,
        train_len=len(train_df),
        val_len=len(val_df),
        test_len=len(test_df)
    )

    metadata_path = artifacts_dir / "preprocessing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved preprocessing metadata JSON to {metadata_path}")
    logger.info("ML Preprocessing Pipeline executed successfully.")


if __name__ == "__main__":
    run_ml_preprocessing_pipeline()
