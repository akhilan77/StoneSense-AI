"""Tabular Data Preprocessor Module for StoneSense-AI (ML).

Handles data cleaning, automatic column type detection, missing value imputation,
outlier detection logging, and sklearn ColumnTransformer pipeline building.
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

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans tabular data according to automated rules."""
        logger.info(f"Initial raw dataset shape: {df.shape}")
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # 1. Remove duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.info(f"Removing {dup_count} duplicate rows...")
            df = df.drop_duplicates().reset_index(drop=True)

        # Detect columns if not done
        if not self.numerical_cols or not self.categorical_cols or not self.target_column:
            self.auto_detect_columns(df)

        # 2. Drop rows with missing target values
        if self.target_column in df.columns:
            target_nulls = df[self.target_column].isnull().sum()
            if target_nulls > 0:
                logger.info(f"Removing {target_nulls} rows with missing target values...")
                df = df.dropna(subset=[self.target_column]).reset_index(drop=True)

        # 3. Detect and log impossible numeric values (e.g. negative physical measures)
        for col in self.numerical_cols:
            if col in df.columns:
                negatives = (df[col] < 0).sum()
                infs = np.isinf(df[col]).sum()
                if negatives > 0:
                    logger.warning(f"Detected {negatives} negative values in feature '{col}'")
                if infs > 0:
                    logger.warning(f"Detected {infs} infinite values in feature '{col}'")

        # 4. Fill missing numerical values with median
        for col in self.numerical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                med_val = df[col].median()
                logger.info(f"Imputing missing values in numerical column '{col}' with median: {med_val}")
                df[col] = df[col].fillna(med_val)

        # 5. Fill missing categorical values with mode
        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                logger.info(f"Imputing missing values in categorical column '{col}' with mode: {mode_val}")
                df[col] = df[col].fillna(mode_val)

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

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits pipeline on X and returns transformed feature array."""
        if self.pipeline is None:
            self.build_pipeline()
        logger.info("Fitting and transforming feature data...")
        transformed = self.pipeline.fit_transform(X)

        # Extract sub-fit components
        if self.numerical_cols and 'num' in self.pipeline.named_transformers_:
            self.scaler = self.pipeline.named_transformers_['num'].named_steps['scaler']
        if self.categorical_cols and 'cat' in self.pipeline.named_transformers_:
            self.encoder = self.pipeline.named_transformers_['cat'].named_steps['encoder']

        return transformed

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
