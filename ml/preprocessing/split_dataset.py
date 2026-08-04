"""Stratified Train / Validation / Test Splitting Module for ML.

Splits dataset into 70% Train, 15% Validation, and 15% Test with stratification.
"""

import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("DatasetSplitter")


def split_tabular_dataset(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataframe into Train (70%), Validation (15%), and Test (15%) sets with stratification.
    
    Args:
        df: Input DataFrame.
        target_column: Name of the target variable for stratification.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting dataset with target column '{target_column}' (70/15/15 ratio)...")
    
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    y = df[target_column]

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    # Second split: Split 30% Temp into 50% Val (15% total) and 50% Test (15% total)
    temp_y = temp_df[target_column]
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_y
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Split sizes - Train: {len(train_df)} ({len(train_df)/len(df):.1%}), "
                f"Val: {len(val_df)} ({len(val_df)/len(df):.1%}), "
                f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})")

    return train_df, val_df, test_df
