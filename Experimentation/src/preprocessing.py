"""
Text Preprocessing Module

This module handles text normalization and tokenization for student answers.
"""

import re
import string


def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and removing extra whitespace.
    
    Args:
        text: Raw input text
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into words, removing punctuation.
    
    Args:
        text: Input text string
        
    Returns:
        List of word tokens
    """
    if not text:
        return []
    
    # Remove punctuation except apostrophes (for contractions)
    text = re.sub(r"[^\w\s']", ' ', text)
    
    # Split on whitespace
    tokens = text.split()
    
    # Remove empty tokens
    tokens = [t.strip() for t in tokens if t.strip()]
    
    return tokens


def preprocess_pipeline(text: str) -> dict:
    """
    Apply full preprocessing pipeline to text.
    
    Args:
        text: Raw input text
        
    Returns:
        Dictionary with normalized text and tokens
    """
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    
    return {
        "original": text,
        "normalized": normalized,
        "tokens": tokens,
        "word_count": len(tokens)
    }
