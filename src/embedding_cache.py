"""
Embedding cache utilities for saving and loading embeddings.

This module provides functions to cache embeddings to disk to avoid
regenerating them on every run (especially slow for GTE).

Includes automatic cache invalidation when source data changes.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import json


def save_embeddings(
    X_train: np.ndarray,
    X_test: np.ndarray,
    embedding_name: str,
    data_size: int,
    cache_dir: Path = Path("data/embeddings_cache")
) -> None:
    """
    Save train and test embeddings to disk with metadata.
    
    Args:
        X_train: Training embeddings
        X_test: Test embeddings
        embedding_name: Name of the embedding (e.g., 'tfidf', 'bert', 'gte')
        data_size: Number of samples in the source dataset
        cache_dir: Directory to save embeddings
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = cache_dir / f"{embedding_name}_train.npy"
    test_path = cache_dir / f"{embedding_name}_test.npy"
    meta_path = cache_dir / f"{embedding_name}_meta.json"
    
    # Save embeddings
    np.save(train_path, X_train)
    np.save(test_path, X_test)
    
    # Save metadata for cache validation
    metadata = {
        'data_size': data_size,
        'train_shape': list(X_train.shape),
        'test_shape': list(X_test.shape)
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"  💾 Cached {embedding_name} embeddings (n={data_size})")


def load_embeddings(
    embedding_name: str,
    current_data_size: int,
    cache_dir: Path = Path("data/embeddings_cache")
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load train and test embeddings from disk if they exist and are valid.
    
    Automatically invalidates cache if data size has changed.
    
    Args:
        embedding_name: Name of the embedding (e.g., 'tfidf', 'bert', 'gte')
        current_data_size: Current number of samples in the dataset
        cache_dir: Directory where embeddings are cached
        
    Returns:
        Tuple of (X_train, X_test) if cache exists and is valid, None otherwise
    """
    train_path = cache_dir / f"{embedding_name}_train.npy"
    test_path = cache_dir / f"{embedding_name}_test.npy"
    meta_path = cache_dir / f"{embedding_name}_meta.json"
    
    # Check if cache exists
    if not (train_path.exists() and test_path.exists() and meta_path.exists()):
        return None
    
    # Validate cache against current data
    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        cached_size = metadata.get('data_size')
        
        # Check if data size has changed
        if cached_size != current_data_size:
            print(f"  ⚠️  Data changed ({cached_size} → {current_data_size})! Clearing cache...")
            clear_cache(cache_dir)
            return None
        
        # Cache is valid - load embeddings
        X_train = np.load(train_path)
        X_test = np.load(test_path)
        print(f"  ✅ Loaded {embedding_name} from cache (n={current_data_size})")
        return X_train, X_test
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠️  Invalid cache metadata! Regenerating...")
        clear_cache(cache_dir)
        return None


def clear_cache(cache_dir: Path = Path("data/embeddings_cache")) -> None:
    """
    Clear all cached embeddings and metadata.
    
    Args:
        cache_dir: Directory where embeddings are cached
    """
    if cache_dir.exists():
        for file in cache_dir.glob("*"):
            if file.is_file():
                file.unlink()
        print(f"  🗑️  Cleared embedding cache")
