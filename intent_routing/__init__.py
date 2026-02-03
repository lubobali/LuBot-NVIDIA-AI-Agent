"""
Intent Routing - 4-tier intent classification system

Components:
- IntentClassifier: 4-tier cascade (regex -> embeddings -> LLM)
- correlation_detector: Deterministic PhD analysis detection
"""

from intent_routing.intent_classifier import IntentClassifier, get_intent_classifier, is_superlative_question
from intent_routing.correlation_detector import detect_phd_analysis_type, is_correlation_query

__all__ = [
    "IntentClassifier",
    "get_intent_classifier",
    "is_superlative_question",
    "detect_phd_analysis_type",
    "is_correlation_query",
]
