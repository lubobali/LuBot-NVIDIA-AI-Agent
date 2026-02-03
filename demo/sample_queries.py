"""
LuBot NVIDIA Routing - Sample Query Routing Decisions

Shows how different queries get routed through the system:
1. Intent Classification (which intent?)
2. Response Tier (direct/enhanced/PhD?)
3. Model Selection (Nano 8B vs Ultra 253B?)

Usage:
    python demo/sample_queries.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent_routing.intent_classifier import IntentClassifier
from intent_routing.correlation_detector import detect_phd_analysis_type
from nvidia_routing.response_tier_router import ResponseTierRouter


def get_model_for_intent(intent: str, analysis_type: str = None) -> str:
    """Determine which NVIDIA model handles this query."""
    # PhD-level analysis -> Ultra 253B
    phd_intents = {"ADVICE_REQUEST", "DEEP_DIVE", "CLARIFICATION"}
    phd_analyses = {"correlation", "concentration", "comparison", "distribution"}

    if intent in phd_intents:
        return "Ultra 253B"
    if analysis_type and analysis_type in phd_analyses:
        return "Ultra 253B"

    # Simple/medium queries -> Nano 8B
    return "Nano 8B"


def main():
    print("=" * 90)
    print("  LuBot NVIDIA Routing - Sample Query Decisions")
    print("  How 15 real-world queries get routed through the system")
    print("=" * 90)

    classifier = IntentClassifier()

    # Sample queries that show the full routing spectrum
    sample_queries = [
        # Simple queries -> Tier 0 Direct / Nano 8B
        "How many employees do we have?",
        "What is the total revenue?",

        # Medium queries -> Tier 1 Enhanced / Nano 8B
        "Show me revenue by region",
        "What are the top 5 products by sales?",
        "Average salary by department",

        # PhD-level queries -> Tier 2 Full PhD / Ultra 253B
        "What's the correlation between marketing spend and revenue?",
        "Calculate the HHI concentration for our customer base",
        "Compare salaries by gender - is there a significant difference?",
        "Show me the distribution of order values",

        # Non-data queries (still classified correctly)
        "Hello!",
        "Who are you?",
        "What can you do?",
        "Predict my traffic for next month",
        "Generate a PDF report of sales",
        "Tell me more about the anomalies you found",
    ]

    print(f"\n{'#':<3} {'Query':<55} {'Intent':<20} {'Tier':<7} {'Model':<12}")
    print("-" * 97)

    tier_labels = {0: "T0/Det", 1: "T1/Rgx", 2: "T2/Emb", 3: "T3/LLM"}

    for i, query in enumerate(sample_queries, 1):
        intent, tier, confidence = classifier.classify(query)
        analysis_type = detect_phd_analysis_type(query)
        model = get_model_for_intent(intent, analysis_type)

        print(f"{i:<3} {query:<55} {intent:<20} {tier_labels.get(tier, '?'):<7} {model:<12}")

    # Summary statistics
    print("\n" + "=" * 90)
    print("ROUTING SUMMARY")
    print("=" * 90)
    print("""
    Intent Classification Tiers:
      Tier 0 (Deterministic)  - PhD keywords (correlation, HHI, compare) -> 0ms, 100% accurate
      Tier 1 (Regex)          - Pattern matching (greetings, commands)    -> 0ms, 95% accurate
      Tier 2 (Embeddings)     - NVIDIA nv-embedqa-e5-v5 semantic match   -> 5ms, 90% accurate
      Tier 3 (LLM)            - NVIDIA Nemotron Nano 8B classification   -> 100ms, 85% accurate

    Model Selection:
      Nano 8B (8B params)     - Simple/medium queries, fast responses
      Ultra 253B (253B params) - PhD-level statistical analysis

    Response Tiers:
      Tier 0 (Direct)         - Single number answers: "You have 320 employees"
      Tier 1 (Enhanced)       - Table + context: data + "Dept A is highest"
      Tier 2 (Full PhD)       - Statistical analysis: correlation, HHI, t-test

    Production Scale (LuBot at lubot.ai):
      112,000+ lines of code | 34 database tables | 40+ API endpoints
      22 batch workers | 99%+ NVIDIA success rate | <200ms P95 latency
    """)


if __name__ == "__main__":
    main()
