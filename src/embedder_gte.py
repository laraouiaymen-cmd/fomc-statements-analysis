"""
GTE (General Text Embeddings) embedding module for text analysis.

This module provides functions to load the Alibaba-NLP GTE-large-en-v1.5 model
and generate high-quality embeddings from text using mean pooling.
"""

from typing import Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def load_gte() -> Tuple[AutoTokenizer, AutoModel]:
    """
    Load GTE tokenizer and model from HuggingFace.
    
    Returns:
        Tuple of (tokenizer, model) where model is set to evaluation mode
    """
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")
    model = AutoModel.from_pretrained(
        "Alibaba-NLP/gte-large-en-v1.5",
        trust_remote_code=True
    )
    model.eval()
    
    return tokenizer, model


def embed_gte(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel
) -> np.ndarray:
    """
    Generate GTE embedding for a single text string.
    
    Uses mean pooling over the last hidden state for better representation.
    
    Args:
        text: Input text string to embed
        tokenizer: GTE tokenizer
        model: GTE model
        
    Returns:
        NumPy array of shape (1024,) containing the mean-pooled embedding
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
    
    # Mean pooling: average over sequence length (excluding padding)
    attention_mask = inputs['attention_mask']
    last_hidden_state = outputs.last_hidden_state
    
    # Expand attention mask to match hidden state dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum embeddings and divide by number of non-padding tokens
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    
    # Convert to numpy array and squeeze to 1D
    return mean_pooled.squeeze().cpu().numpy()


def embed_many_gte(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 8
) -> np.ndarray:
    """
    Generate GTE embeddings for multiple text strings with batch processing.
    
    Optimized for CPU inference using batches to reduce overhead.
    Uses mean pooling over the last hidden state for better representation.
    
    Args:
        texts: List of input text strings to embed
        tokenizer: GTE tokenizer
        model: GTE model
        batch_size: Number of texts to process at once (default 8, optimal for CPU)
        
    Returns:
        NumPy array of shape (N, 1024) where N is the number of texts
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
        
        # Mean pooling for each text in batch
        attention_mask = inputs['attention_mask']
        last_hidden_state = outputs.last_hidden_state
        
        # Expand attention mask to match hidden state dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum embeddings and divide by number of non-padding tokens
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Convert to numpy
        batch_embeddings = mean_pooled.cpu().numpy()
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    return np.vstack(embeddings)
