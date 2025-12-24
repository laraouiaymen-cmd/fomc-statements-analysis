"""
TF-IDF embedding module for text vectorization.

This module provides functions to build TF-IDF representations of text data
using scikit-learn's TfidfVectorizer.
"""

from typing import Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    train_texts: list[str],
    test_texts: list[str],
    max_features: int | None = None,
    vocab_ratio: float = 0.35,
    min_df: int = 2
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Build TF-IDF representations for train and test text data.
    
    The vectorizer is fit on the training data only, then used to transform
    both training and test data. This prevents data leakage.
    
    If max_features is None, it will be dynamically calculated as a ratio
    of the total vocabulary size. The min_df parameter filters out rare words
    that appear in too few documents.
    
    Args:
        train_texts: List of training text strings
        test_texts: List of test text strings
        max_features: Maximum number of features (top words by importance).
                     If None, calculated as vocab_ratio * total_vocabulary
        vocab_ratio: Ratio of vocabulary to use when max_features is None (default 0.35)
        min_df: Minimum number of documents a word must appear in to be included (default 2)
        
    Returns:
        Tuple of (X_train_tfidf, X_test_tfidf, vectorizer) where:
            - X_train_tfidf: NumPy array of shape (N_train, n_features)
            - X_test_tfidf: NumPy array of shape (N_test, n_features)
            - vectorizer: Fitted TfidfVectorizer instance
    """
    # Dynamically determine max_features if not specified
    if max_features is None:
        # First pass to count vocabulary (after min_df filtering)
        temp_vectorizer = TfidfVectorizer(min_df=min_df)
        temp_vectorizer.fit(train_texts)
        total_vocab = len(temp_vectorizer.vocabulary_)
        
        # Calculate max_features as ratio of filtered vocabulary
        max_features = int(total_vocab * vocab_ratio)
    
    # Create TF-IDF vectorizer with determined parameters
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    
    # Fit on training data only
    X_train_tfidf_sparse = vectorizer.fit_transform(train_texts)
    
    # Transform test data using fitted vectorizer
    X_test_tfidf_sparse = vectorizer.transform(test_texts)
    
    # Convert sparse matrices to dense numpy arrays
    X_train_tfidf = X_train_tfidf_sparse.toarray()
    X_test_tfidf = X_test_tfidf_sparse.toarray()
    
    return X_train_tfidf, X_test_tfidf, vectorizer
