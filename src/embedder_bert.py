"""
BERT embedding module for general text analysis.

This module provides functions to load the BERT model and generate
embeddings from text using the CLS token representation.
"""

from typing import Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def load_bert() -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load BERT tokenizer and model from HuggingFace.
    
    Returns:
        Tuple of (tokenizer, model) where model is set to evaluation mode
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    
    return tokenizer, model


def embed_bert(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel
) -> np.ndarray:
    """
    Generate BERT embedding for a single text string.
    
    Extracts the CLS token representation from the last hidden state.
    
    Args:
        text: Input text string to embed
        tokenizer: BERT tokenizer
        model: BERT model
        
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


def embed_many_bert(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 8
) -> np.ndarray:
    """
    Generate BERT embeddings for multiple text strings with batch processing.
    
    Optimized for CPU inference using batches to reduce overhead.
    
    Args:
        texts: List of input text strings to embed
        tokenizer: BERT tokenizer
        model: BERT model
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
