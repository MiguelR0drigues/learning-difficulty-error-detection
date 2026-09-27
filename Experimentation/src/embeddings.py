"""
Semantic Embedding Module

This module handles semantic representation of texts using Sentence-BERT
for computing similarity between reference and student answers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# Global model instance (lazy loading)
_model = None


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load the Sentence-BERT model (cached after first load).
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        Loaded SentenceTransformer model
    """
    global _model
    if _model is None:
        print(f"Loading Sentence-BERT model: {model_name}...")
        _model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
    return _model


def get_embedding(text: str, model: SentenceTransformer = None) -> np.ndarray:
    """
    Generate embedding vector for a text.
    
    Args:
        text: Input text string
        model: Optional pre-loaded model (uses global model if None)
        
    Returns:
        Numpy array embedding vector
    """
    if model is None:
        model = load_model()
    
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def compute_similarity(text1: str, text2: str, model: SentenceTransformer = None) -> float:
    """
    Compute cosine similarity between two texts.
    
    Args:
        text1: First text string
        text2: Second text string
        model: Optional pre-loaded model
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    if model is None:
        model = load_model()
    
    # Get embeddings
    emb1 = get_embedding(text1, model)
    emb2 = get_embedding(text2, model)
    
    # Compute cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    # Ensure result is between 0 and 1
    return max(0.0, min(1.0, float(similarity)))


def get_embeddings_batch(texts: list[str], model: SentenceTransformer = None) -> np.ndarray:
    """
    Generate embeddings for multiple texts efficiently.
    
    Args:
        texts: List of text strings
        model: Optional pre-loaded model
        
    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    if model is None:
        model = load_model()
    
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings
