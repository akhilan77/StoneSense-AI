"""Tabular Data Preprocessor Module for StoneSense-AI (ML).

Handles data cleaning, automatic column type detection, duplicate checking (failing loudly),
class balance verification, missing value imputation, and sklearn ColumnTransformer pipeline building.
"""

from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger("TabularPreprocessor")


def check_duplicate_rows(df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    """Checks for duplicate rows in the dataset and fails loudly unless drop_duplicates is True.
    
    Args:
        df: Input DataFrame.
        drop_duplicates: If True, logs and drops duplicate rows. If False and duplicates exist,
            raises ValueError with a full diagnostic report of duplicate rows.
            
    Returns:
        DataFrame with duplicates handled.
        
    Raises:
        ValueError: If duplicate rows are detected and drop_duplicates is False.
    """
    clean_cols = [c for c in df.columns if c != 'Unnamed: 0']
    df_check = df[clean_cols]
    duplicate_mask = df_check.duplicated(keep=False)
    num_duplicates = int(duplicate_mask.sum())

    if num_duplicates > 0:
        dup_rows = df[duplicate_mask]
        dup_report = (
            f"Data integrity error: Found {num_duplicates} duplicate rows in dataset "
            f"(unique duplicate patterns: {df_check.duplicated().sum()}).\n"
            f"Duplicate row indices: {list(dup_rows.index)}\n"
            f"Duplicate rows preview:\n{dup_rows.to_string()}"
        )
        if not drop_duplicates:
            logger.error(dup_report)
            raise ValueError(dup_report)
        else:
            logger.warning(f"Dropping {df_check.duplicated().sum()} duplicate rows:\n{dup_rows.to_string()}")
            df = df.drop_duplicates(subset=clean_cols).reset_index(drop=True)
            logger.info(f"Dataset shape after duplicate removal: {df.shape}")

    return df


def check_class_balance(
    df: pd.DataFrame,
    target_column: str,
    min_samples_per_class: int = 5,
    max_imbalance_ratio: float = 10.0
) -> Dict[str, Any]:
    """Verifies class representation in tabular target column and fails loudly if invalid.
    
    Args:
        df: Input DataFrame.
        target_column: Name of target column.
        min_samples_per_class: Minimum sample threshold required per class.
        max_imbalance_ratio: Maximum allowed ratio between majority and minority classes.
        
    Returns:
        Dictionary containing class balance summary.
        
    Raises:
        ValueError: If class count < min_samples_per_class or imbalance exceeds max_imbalance_ratio.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    counts = df[target_column].value_counts().to_dict()
    total = len(df)

    if len(counts) < 2:
        raise ValueError(
            f"Class balance error: Target column '{target_column}' contains only {len(counts)} class ({counts}). "
            f"Binary classification requires at least 2 distinct classes."
        )

    min_count = min(counts.values())
    max_count = max(counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    if min_count < min_samples_per_class:
        raise ValueError(
            f"Class balance error: Minority class has only {min_count} samples, which is below "
            f"the required threshold of {min_samples_per_class}. Distribution: {counts}"
        )

    if imbalance_ratio > max_imbalance_ratio:
        raise ValueError(
            f"Class balance error: Severe class imbalance detected (ratio: {imbalance_ratio:.2f}:1 > {max_imbalance_ratio}:1). "
            f"Distribution: {counts}"
        )

    summary = {
        "target_column": target_column,
        "class_counts": counts,
        "class_proportions": {k: round(v / total, 4) for k, v in counts.items()},
        "total_samples": total,
        "imbalance_ratio": round(imbalance_ratio, 2)
    }
    logger.info(f"Class balance check passed: {summary}")
    return summary


class TabularPreprocessor:
    """Preprocessor for tabular machine learning datasets."""

    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.pipeline: Optional[ColumnTransformer] = None
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[OneHotEncoder] = None

    def auto_detect_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
        """Automatically detects numerical, categorical, and target columns."""
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Detect target column if not explicitly given
        if self.target_column is None:
            for candidate in ['target', 'Class', 'label', 'diag']:
                if candidate in df.columns:
                    self.target_column = candidate
                    break
            if self.target_column is None:
                self.target_column = df.columns[-1]

        logger.info(f"Detected target column: '{self.target_column}'")

        feature_cols = [c for c in df.columns if c != self.target_column]
        
        self.numerical_cols = list(df[feature_cols].select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(df[feature_cols].select_dtypes(include=['object', 'category']).columns)

        logger.info(f"Detected numerical features ({len(self.numerical_cols)}): {self.numerical_cols}")
        logger.info(f"Detected categorical features ({len(self.categorical_cols)}): {self.categorical_cols}")

        return self.numerical_cols, self.categorical_cols, self.target_column

    def clean_data(self, df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
        """Cleans tabular data according to automated rules, verifying integrity."""
        logger.info(f"Initial raw dataset shape: {df.shape}")
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # 1. Duplicate check (fails loudly if drop_duplicates=False and duplicates exist)
        df = check_duplicate_rows(df, drop_duplicates=drop_duplicates)

        # Detect columns if not done
        if not self.numerical_cols or not self.categorical_cols or not self.target_column:
            self.auto_detect_columns(df)

        # 2. Drop rows with missing target values
        if self.target_column in df.columns:
            target_nulls = df[self.target_column].isnull().sum()
            if target_nulls > 0:
                logger.info(f"Removing {target_nulls} rows with missing target values...")
                df = df.dropna(subset=[self.target_column]).reset_index(drop=True)

            # 3. Class balance verification
            check_class_balance(df, target_column=self.target_column)

        # 4. Detect and log impossible numeric values (e.g. negative physical measures)
        for col in self.numerical_cols:
            if col in df.columns:
                negatives = (df[col] < 0).sum()
                infs = np.isinf(df[col]).sum()
                if negatives > 0:
                    logger.warning(f"Detected {negatives} negative values in feature '{col}'")
                if infs > 0:
                    logger.warning(f"Detected {infs} infinite values in feature '{col}'")

        logger.info(f"Cleaned dataset shape: {df.shape}")
        return df

    def build_pipeline(self) -> ColumnTransformer:
        """Constructs sklearn ColumnTransformer preprocessing pipeline."""
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers = []
        if self.numerical_cols:
            transformers.append(('num', num_transformer, self.numerical_cols))
        if self.categorical_cols:
            transformers.append(('cat', cat_transformer, self.categorical_cols))

        self.pipeline = ColumnTransformer(transformers=transformers, remainder='drop')
        return self.pipeline

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        """Fits preprocessing pipeline strictly on training data."""
        if self.pipeline is None:
            self.build_pipeline()
        logger.info("Fitting preprocessing pipeline on training data only...")
        self.pipeline.fit(X)

        # Extract sub-fit components
        if self.numerical_cols and 'num' in self.pipeline.named_transformers_:
            self.scaler = self.pipeline.named_transformers_['num'].named_steps['scaler']
        if self.categorical_cols and 'cat' in self.pipeline.named_transformers_:
            self.encoder = self.pipeline.named_transformers_['cat'].named_steps['encoder']

        return self

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits pipeline on training data X and returns transformed feature array."""
        self.fit(X)
        return self.pipeline.transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature data X using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline has not been fitted yet!")
        return self.pipeline.transform(X)

    def save_artifacts(self, artifact_dir: Path) -> Dict[str, Path]:
        """Saves scaler, encoder, feature columns, and full pipeline to disk."""
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        if self.pipeline is not None:
            pipe_path = artifact_dir / "preprocessing_pipeline.pkl"
            joblib.dump(self.pipeline, pipe_path)
            saved_paths["pipeline"] = pipe_path

        if self.scaler is not None:
            scaler_path = artifact_dir / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            saved_paths["scaler"] = scaler_path

        if self.encoder is not None:
            encoder_path = artifact_dir / "encoder.pkl"
            joblib.dump(self.encoder, encoder_path)
            saved_paths["encoder"] = encoder_path

        cols_path = artifact_dir / "feature_columns.pkl"
        all_feature_cols = self.numerical_cols + self.categorical_cols
        joblib.dump(all_feature_cols, cols_path)
        saved_paths["feature_columns"] = cols_path

        logger.info(f"Successfully saved preprocessing artifacts to {artifact_dir}")
        return saved_paths
