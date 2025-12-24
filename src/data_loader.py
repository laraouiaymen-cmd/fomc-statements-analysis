"""
Data loading and preprocessing module for FOMC dataset.

This module provides functions to load, split, and prepare FOMC statement data
for machine learning tasks. All paths are parameterized for maximum reusability.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_fomc_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load FOMC dataset from CSV file and perform basic preprocessing.
    
    Args:
        csv_path: Path to the CSV file containing FOMC statements
        
    Returns:
        DataFrame with cleaned and sorted FOMC data
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        ValueError: If required columns are missing
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime (format: YYYYMMDD as integer)
    df["date"] = pd.to_datetime(df["date"], format='%Y%m%d')
    
    # Sort chronologically (oldest to newest)
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically into train and test sets.
    
    This function does NOT shuffle the data, preserving temporal order.
    
    Args:
        df: DataFrame to split
        train_ratio: Proportion of data to use for training (default: 0.70)
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        ValueError: If train_ratio is not between 0 and 1
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    
    # Compute split index
    train_end = int(train_ratio * len(df))
    
    # Split without shuffling
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()
    
    return train_df, test_df


def get_X_y(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[
    list[str], list[str],
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray
]:
    """
    Extract features (text) and labels from train and test DataFrames.
    
    This function does NOT perform any tokenization or embedding.
    It only extracts raw text and both regression and classification labels.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class) where:
            - X_train: List of raw text strings from training set
            - X_test: List of raw text strings from test set
            - y_train_reg: NumPy array of training regression targets (next_day_return)
            - y_test_reg: NumPy array of test regression targets (next_day_return)
            - y_train_class: NumPy array of training classification targets (label)
            - y_test_class: NumPy array of test classification targets (label)
            
    Raises:
        KeyError: If required columns are missing from DataFrames
    """
    # Extract raw text as lists
    X_train = train_df["clean_text"].tolist()
    X_test = test_df["clean_text"].tolist()
    
    # Extract regression targets as numpy arrays
    y_train_reg = train_df["next_day_return"].values
    y_test_reg = test_df["next_day_return"].values
    
    # Extract classification targets as numpy arrays
    y_train_class = train_df["label"].values
    y_test_class = test_df["label"].values
    
    return X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class

    