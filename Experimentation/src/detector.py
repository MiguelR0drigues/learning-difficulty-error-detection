"""
Error Pattern Detection Module

This module implements heuristic-based rules for detecting error patterns
in student answers, with confidence scoring and explanations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    """Result of error pattern detection."""
    predicted_error_type: str
    confidence: float
    explanation: str
    details: dict


# Configurable thresholds for heuristic rules
THRESHOLDS = {
    "similarity_high": 0.75,             # Above this → likely correct
    "similarity_conceptual": 0.55,       # Below this → conceptual error
    "min_word_count": 5,                 # Below this → vague answer
    "keyword_coverage_threshold": 0.4,   # Below this → missing concepts
    "spelling_errors_threshold": 3,      # At or above → linguistic error
}


def detect_error(
    reference: str,
    student: str,
    similarity: float,
    features: dict
) -> DetectionResult:
    """
    Detect error pattern in a student answer using heuristic rules.
    
    The detection follows this priority order:
    1. Check for vague/superficial answers (too short)
    2. Check for linguistic errors (spelling/grammar)
    3. Check for conceptual errors (low semantic similarity)
    4. Check for missing key concepts
    5. If none of the above, classify as correct
    
    Args:
        reference: Reference answer text
        student: Student answer text
        similarity: Semantic similarity score (0-1)
        features: Feature dictionary from features.compute_features()
        
    Returns:
        DetectionResult with predicted error type, confidence, and explanation
    """
    word_count = features.get("word_count", 0)
    num_missing_keywords = features.get("num_missing_keywords", 0)
    missing_keywords = features.get("missing_keywords", [])
    num_spelling_errors = features.get("num_spelling_errors", 0)
    misspelled_words = features.get("misspelled_words", [])
    keyword_coverage = features.get("keyword_coverage", 1.0)
    
    # Track all detected issues for combined analysis
    issues = []
    
    # First check if answer is likely correct (high similarity + good keyword coverage)
    if similarity >= THRESHOLDS["similarity_high"] and keyword_coverage >= 0.5:
        confidence = min(1.0, similarity * keyword_coverage)
        return DetectionResult(
            predicted_error_type="none",
            confidence=confidence,
            explanation="Answer demonstrates good understanding with adequate coverage of key concepts.",
            details={
                "similarity": similarity,
                "word_count": word_count,
                "keyword_coverage": keyword_coverage
            }
        )
    
    # Rule 1: Vague answer (too short) - highest priority for very short answers
    if word_count < THRESHOLDS["min_word_count"]:
        confidence = min(1.0, (THRESHOLDS["min_word_count"] - word_count) / THRESHOLDS["min_word_count"] + 0.5)
        issues.append({
            "type": "vague_answer",
            "confidence": confidence,
            "explanation": f"Response is too short ({word_count} words). Lacks sufficient detail to demonstrate understanding."
        })
    
    # Rule 2: Linguistic errors (spelling) - high priority
    if num_spelling_errors >= THRESHOLDS["spelling_errors_threshold"]:
        confidence = min(1.0, num_spelling_errors / 5 + 0.5)
        issues.append({
            "type": "linguistic_error",
            "confidence": confidence,
            "explanation": f"Multiple spelling errors detected: {', '.join(misspelled_words[:5])}."
        })
    
    # Rule 3: Conceptual error (very low semantic similarity) - indicates wrong understanding
    if similarity < THRESHOLDS["similarity_conceptual"]:
        confidence = min(1.0, (THRESHOLDS["similarity_conceptual"] - similarity) / THRESHOLDS["similarity_conceptual"] + 0.5)
        issues.append({
            "type": "conceptual_error",
            "confidence": confidence,
            "explanation": f"Low semantic similarity ({similarity:.2f}) indicates incorrect or confused understanding of the concept."
        })
    
    # Rule 4: Missing key concepts - based on keyword coverage ratio
    if keyword_coverage < THRESHOLDS["keyword_coverage_threshold"] and similarity >= THRESHOLDS["similarity_conceptual"]:
        confidence = min(1.0, (THRESHOLDS["keyword_coverage_threshold"] - keyword_coverage) / THRESHOLDS["keyword_coverage_threshold"] + 0.4)
        issues.append({
            "type": "missing_key_concepts",
            "confidence": confidence,
            "explanation": f"Low keyword coverage ({keyword_coverage:.0%}). Missing key concepts: {', '.join(list(missing_keywords)[:5])}."
        })
    
    # Determine the primary error type
    if not issues:
        # No issues detected - answer is correct (moderate similarity)
        confidence = max(0.5, similarity)
        return DetectionResult(
            predicted_error_type="none",
            confidence=confidence,
            explanation="Answer demonstrates understanding of the concept.",
            details={
                "similarity": similarity,
                "word_count": word_count,
                "keyword_coverage": keyword_coverage
            }
        )
    
    # Sort issues by confidence and return the most confident prediction
    issues.sort(key=lambda x: x["confidence"], reverse=True)
    primary_issue = issues[0]
    
    # Build detailed explanation including secondary issues
    explanation = primary_issue["explanation"]
    if len(issues) > 1:
        secondary_types = [i["type"] for i in issues[1:3]]
        explanation += f" Additionally detected: {', '.join(secondary_types)}."
    
    return DetectionResult(
        predicted_error_type=primary_issue["type"],
        confidence=primary_issue["confidence"],
        explanation=explanation,
        details={
            "similarity": similarity,
            "word_count": word_count,
            "keyword_coverage": keyword_coverage,
            "all_issues": issues
        }
    )


def result_to_dict(result: DetectionResult) -> dict:
    """
    Convert DetectionResult to dictionary format for JSON output.
    
    Args:
        result: DetectionResult instance
        
    Returns:
        Dictionary representation
    """
    return {
        "predicted_error_type": result.predicted_error_type,
        "confidence": round(result.confidence, 2),
        "explanation": result.explanation
    }
