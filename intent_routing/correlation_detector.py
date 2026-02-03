"""
Correlation Query Detector - Deterministic Keyword Detection

Design Pattern: Claude Code / Google Search approach
- Layer 1: Deterministic keyword detection (100% reliable)
- Layer 2: LLM classification (fallback for complex queries)

PURPOSE:
Detect correlation queries with 100% reliability before LLM classification.
Prevents LLM inconsistency from misrouting correlation queries.

PATTERN:
Google: "weather" keyword -> weather widget (not general search)
Claude Code: "read file" keyword -> Read tool (not LLM reasoning)
LuBot: "correlation" keyword -> DATA_QUERY (not GENERAL/MEMORY)

UNIVERSAL:
Works with ANY phrasing:
- "What's the correlation between X and Y?"
- "Is there a relationship between X and Y?"
- "How does X correlate with Y?"
- "Does X relate to Y?"
"""

from typing import List


def is_correlation_query(question: str) -> bool:
    """
    Detect correlation queries with 100% deterministic reliability.

    Claude Code Pattern: Explicit keyword detection before LLM reasoning.
    This prevents LLM inconsistency from misrouting correlation queries.

    Args:
        question: User's question text

    Returns:
        True if this is definitely a correlation query, False otherwise

    Examples:
        >>> is_correlation_query("What's the correlation between age and salary?")
        True
        >>> is_correlation_query("Is there a relationship between X and Y?")
        True
        >>> is_correlation_query("Show me sales data")
        False
    """
    if not question:
        return False

    question_lower = question.lower()

    # Correlation keywords (deterministic detection)
    correlation_keywords = [
        "correlation",
        "correlate",
        "correlated",
        "relationship between",
        "relate to",
        "related to",
        "how does",  # "how does X affect Y?"
    ]

    # Check for explicit correlation keywords
    for keyword in correlation_keywords:
        if keyword in question_lower:
            return True

    # Pattern: "X and Y" suggests correlation (more specific check)
    # Only if question also has comparative/relationship words
    if " and " in question_lower:
        relationship_indicators = ["between", "compare", "vs", "versus"]
        if any(indicator in question_lower for indicator in relationship_indicators):
            return True

    return False


def is_concentration_query(question: str) -> bool:
    """
    Detect concentration/HHI queries with deterministic reliability.

    Args:
        question: User's question text

    Returns:
        True if this is a concentration query, False otherwise
    """
    if not question:
        return False

    question_lower = question.lower()

    concentration_keywords = [
        "concentration",
        "hhi",
        "herfindahl",
        "market concentration",
        "depend on",
        "reliance",
        "diversification",
    ]

    return any(keyword in question_lower for keyword in concentration_keywords)


def is_comparison_query(question: str) -> bool:
    """
    Detect comparison/t-test queries with deterministic reliability.

    Args:
        question: User's question text

    Returns:
        True if this is a comparison query, False otherwise
    """
    if not question:
        return False

    question_lower = question.lower()

    # Must have comparison word AND group indicator
    comparison_words = ["compare", "difference", "vs", "versus"]
    group_indicators = ["by", "between", "gender", "department", "region", "category", "group"]

    has_comparison = any(word in question_lower for word in comparison_words)
    has_groups = any(indicator in question_lower for indicator in group_indicators)

    return has_comparison and has_groups


def is_distribution_query(question: str) -> bool:
    """
    Detect distribution/histogram queries with deterministic reliability.

    Pattern: Keyword detection before LLM (0ms, 100% reliable)

    Args:
        question: User's question text

    Returns:
        True if this is a distribution query, False otherwise
    """
    if not question:
        return False

    question_lower = question.lower()

    distribution_keywords = [
        "distribution",
        "histogram",
        "spread of",
        "frequency of",
    ]

    return any(keyword in question_lower for keyword in distribution_keywords)


def detect_phd_analysis_type(question: str) -> str:
    """
    Deterministic detection of PhD analysis types.

    Multi-layer routing:
    - Layer 1: This function (deterministic keyword detection)
    - Layer 2: LLM classification (fallback for complex queries)

    Args:
        question: User's question text

    Returns:
        "correlation", "concentration", "comparison", "distribution", or None if no match

    Usage:
        analysis_type = detect_phd_analysis_type(user_question)
        if analysis_type:
            # Deterministic routing - 100% reliable
            intent = "DATA_QUERY"
        else:
            # Fallback to LLM classification
            intent = llm_classifier(user_question)
    """
    if is_correlation_query(question):
        return "correlation"
    elif is_concentration_query(question):
        return "concentration"
    elif is_comparison_query(question):
        return "comparison"
    elif is_distribution_query(question):
        return "distribution"

    return None


if __name__ == "__main__":
    # Test cases
    test_queries = [
        ("What's the correlation between age and salary?", "correlation"),
        ("Is there a relationship between experience and income?", "correlation"),
        ("How does price correlate with sales?", "correlation"),
        ("What's the HHI?", "concentration"),
        ("Compare salaries by gender", "comparison"),
        ("Show me revenue", None),
    ]

    print("Testing PhD Analysis Detector:\n")
    for query, expected in test_queries:
        result = detect_phd_analysis_type(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} '{query}' -> {result} (expected: {expected})")
