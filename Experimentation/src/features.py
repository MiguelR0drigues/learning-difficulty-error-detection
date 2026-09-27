"""
Feature Extraction Module

This module extracts features from student answers for error detection,
including keyword matching, spelling errors, and word counts.
"""

from spellchecker import SpellChecker
from .preprocessing import tokenize, normalize_text


# Common stop words to exclude from keyword matching
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'this', 'that', 'these', 'those', 'what',
    'which', 'who', 'whom', 'it', 'its', 'they', 'them', 'their', 'we',
    'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my'
}


def count_words(text: str) -> int:
    """
    Count the number of words in a text.
    
    Args:
        text: Input text string
        
    Returns:
        Number of words
    """
    tokens = tokenize(normalize_text(text))
    return len(tokens)


def extract_keywords(reference: str, min_length: int = 3) -> set[str]:
    """
    Extract key content words from reference answer.
    
    Args:
        reference: Reference answer text
        min_length: Minimum word length to consider as keyword
        
    Returns:
        Set of keywords
    """
    tokens = tokenize(normalize_text(reference))
    
    # Filter out stop words and short words
    keywords = {
        token for token in tokens
        if token not in STOP_WORDS and len(token) >= min_length
    }
    
    return keywords


def find_missing_keywords(reference: str, student: str, min_length: int = 3) -> dict:
    """
    Find keywords from reference that are missing in student answer.
    
    Args:
        reference: Reference answer text
        student: Student answer text
        min_length: Minimum word length for keywords
        
    Returns:
        Dictionary with keyword analysis
    """
    ref_keywords = extract_keywords(reference, min_length)
    student_tokens = set(tokenize(normalize_text(student)))
    
    # Find which reference keywords appear in student answer
    found_keywords = ref_keywords & student_tokens
    missing_keywords = ref_keywords - student_tokens
    
    # Calculate coverage ratio
    coverage = len(found_keywords) / len(ref_keywords) if ref_keywords else 1.0
    
    return {
        "reference_keywords": ref_keywords,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "coverage_ratio": coverage,
        "num_missing": len(missing_keywords)
    }


def count_spelling_errors(text: str) -> dict:
    """
    Count spelling errors in text using pyspellchecker.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with spelling analysis
    """
    spell = SpellChecker()
    tokens = tokenize(normalize_text(text))
    
    # Filter out very short words and find misspelled words
    words_to_check = [t for t in tokens if len(t) > 2]
    misspelled = spell.unknown(words_to_check)
    
    # Get suggestions for misspelled words
    corrections = {}
    for word in misspelled:
        correction = spell.correction(word)
        if correction and correction != word:
            corrections[word] = correction
    
    return {
        "total_words": len(words_to_check),
        "misspelled_words": list(misspelled),
        "num_errors": len(misspelled),
        "suggested_corrections": corrections,
        "error_ratio": len(misspelled) / len(words_to_check) if words_to_check else 0.0
    }


def compute_features(reference: str, student: str) -> dict:
    """
    Compute all features for a student answer.
    
    Args:
        reference: Reference answer text
        student: Student answer text
        
    Returns:
        Combined feature dictionary
    """
    word_count = count_words(student)
    keyword_analysis = find_missing_keywords(reference, student)
    spelling_analysis = count_spelling_errors(student)
    
    return {
        "word_count": word_count,
        "keyword_coverage": keyword_analysis["coverage_ratio"],
        "num_missing_keywords": keyword_analysis["num_missing"],
        "missing_keywords": list(keyword_analysis["missing_keywords"]),
        "num_spelling_errors": spelling_analysis["num_errors"],
        "misspelled_words": spelling_analysis["misspelled_words"],
        "spelling_error_ratio": spelling_analysis["error_ratio"]
    }
