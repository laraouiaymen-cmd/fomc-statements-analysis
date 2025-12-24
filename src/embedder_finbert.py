"""
FinBERT embedding module for financial text analysis.

This module provides functions to load the FinBERT model and generate
embeddings from text using the CLS token representation.
"""

from typing import Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def load_finbert() -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load FinBERT tokenizer and model from HuggingFace.
    
    Returns:
        Tuple of (tokenizer, model) where model is set to evaluation mode
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.eval()
    
    return tokenizer, model


def embed_finbert(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel
) -> np.ndarray:
    """
    Generate FinBERT embedding for a single text string.
    
    Extracts the CLS token representation from the last hidden state.
    
    Args:
        text: Input text string to embed
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        
    Returns:
        NumPy array of shape (768,) containing the CLS token embedding
    """
    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Generate embeddings without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract CLS token (first token of first sequence)
    cls_embedding = outputs.last_hidden_state[0, 0, :]
    
    # Convert to numpy array
    return cls_embedding.cpu().numpy()


def embed_many_finbert(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 8
) -> np.ndarray:
    """
    Generate FinBERT embeddings for multiple text strings with batch processing.
    
    Optimized for CPU inference using batches to reduce overhead.
    
    Args:
        texts: List of input text strings to embed
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        batch_size: Number of texts to process at once (default 8, optimal for CPU)
        
    Returns:
        NumPy array of shape (N, 768) where N is the number of texts
    """
    embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Generate embeddings without gradient computation
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract CLS token for each text in batch
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    return np.vstack(embeddings)
