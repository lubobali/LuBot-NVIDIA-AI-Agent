"""
LuBot NVIDIA Routing - Quick Start Demo

Prerequisites:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    pip install -r requirements.txt

Usage:
    python demo/quickstart.py
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(message)s")


def check_api_key():
    """Verify NVIDIA API key is set."""
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print("=" * 60)
        print("ERROR: NVIDIA_API_KEY not set")
        print()
        print("Get your free key at: https://build.nvidia.com")
        print("Then run: export NVIDIA_API_KEY='nvapi-your-key'")
        print("=" * 60)
        return False
    print(f"NVIDIA API key found: {key[:12]}...")
    return True


def demo_nvidia_client():
    """Demo 1: Direct NVIDIA API call."""
    print("\n" + "=" * 60)
    print("DEMO 1: Direct NVIDIA Client")
    print("Model: Nemotron Nano 8B (fast, cost-effective)")
    print("=" * 60)

    from nvidia_routing.nvidia_client import NVIDIAClient

    client = NVIDIAClient()

    # Tier 1: Simple query -> Nano 8B
    response = client.chat_tier1(
        messages=[{"role": "user", "content": "What are the top 3 metrics a business should track for revenue growth? Be concise."}]
    )

    if response.success:
        print(f"\nModel: {response.model}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Response:\n{response.content}")
    else:
        print(f"\nError: {response.error}")

    return response.success


def demo_llm_router():
    """Demo 2: Multi-model routing with failover."""
    print("\n" + "=" * 60)
    print("DEMO 2: LLM Router (NVIDIA primary + Groq fallback)")
    print("=" * 60)

    from nvidia_routing.llm_router import get_llm_router

    router = get_llm_router()

    # Tier 1: Simple query -> Nano 8B
    print("\n--- Tier 1 (Nano 8B): Simple query ---")
    response = router.chat_completion(
        tier=1,
        messages=[{"role": "user", "content": "Summarize what HHI (Herfindahl-Hirschman Index) measures in one sentence."}],
        max_tokens=100
    )
    print(f"Model: {response.model}")
    print(f"Fallback used: {response.used_fallback}")
    print(f"Response: {response.content}")

    # Tier 2: Complex query -> Ultra 253B
    print("\n--- Tier 2 (Ultra 253B): PhD-level analysis ---")
    response = router.chat_completion(
        tier=2,
        messages=[{"role": "user", "content": "Explain Simpson's Paradox with a concrete business example where aggregate data shows one trend but segmented data reveals the opposite."}],
        max_tokens=300
    )
    print(f"Model: {response.model}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Response: {response.content}")


def demo_embeddings():
    """Demo 3: NVIDIA embeddings."""
    print("\n" + "=" * 60)
    print("DEMO 3: NVIDIA Embeddings (nv-embedqa-e5-v5)")
    print("Dimensions: 1024 | SentenceTransformer-compatible")
    print("=" * 60)

    from nvidia_routing.nvidia_embeddings import NVIDIAEmbeddings
    import numpy as np

    model = NVIDIAEmbeddings()

    # Encode sample texts
    texts = [
        "Show me revenue by region",
        "What's the correlation between price and sales?",
        "Hello, how are you?",
    ]

    embeddings = model.encode(texts)
    print(f"\nEncoded {len(texts)} texts -> shape: {embeddings.shape}")

    # Show cosine similarities
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            print(f"Similarity('{texts[i][:40]}...', '{texts[j][:40]}...'): {sim:.3f}")


def demo_intent_classification():
    """Demo 4: 4-tier intent classification."""
    print("\n" + "=" * 60)
    print("DEMO 4: Intent Classification (4-tier cascade)")
    print("Tier 0: Deterministic | Tier 1: Regex | Tier 2: Embeddings | Tier 3: LLM")
    print("=" * 60)

    from intent_routing.intent_classifier import IntentClassifier

    classifier = IntentClassifier()

    test_queries = [
        "What's the correlation between age and salary?",
        "Hello!",
        "Show me revenue by region",
        "Who are you?",
        "What's the HHI concentration?",
        "Compare salaries by department",
        "Predict my traffic for next month",
        "Generate a PDF report",
        "Tell me more about the anomalies",
        "What can you do?",
    ]

    print(f"\n{'Query':<50} {'Intent':<20} {'Tier':<6} {'Conf':<6}")
    print("-" * 82)

    for query in test_queries:
        intent, tier, confidence = classifier.classify(query)
        tier_labels = {0: "T0/Det", 1: "T1/Rgx", 2: "T2/Emb", 3: "T3/LLM"}
        print(f"{query:<50} {intent:<20} {tier_labels.get(tier, '?'):<6} {confidence:.2f}")


def demo_response_tier():
    """Demo 5: Response tier routing."""
    print("\n" + "=" * 60)
    print("DEMO 5: Response Tier Router")
    print("Tier 0: Direct | Tier 1: Enhanced | Tier 2: Full PhD")
    print("=" * 60)

    from nvidia_routing.response_tier_router import ResponseTierRouter

    test_cases = [
        ({"row_count": 1, "data": [{"count": 320}]}, "user_generic_count", "basic"),
        ({"row_count": 5, "data": [{"dept": "HR", "avg": 75000}]}, "user_generic_avg_by_dimension", "basic"),
        ({"row_count": 2, "data": [{"x": 1, "y": 2}]}, "user_generic_avg_by_dimension", "correlation"),
        ({"row_count": 15, "data": [{}] * 15}, "user_generic_total_by_dimension", "basic"),
    ]

    for sql_data, template, analysis_type in test_cases:
        tier = ResponseTierRouter.get_response_tier(sql_data, template, analysis_type)
        print(f"  rows={sql_data['row_count']:<3} template={template:<35} analysis={analysis_type:<12} -> {tier}")

    # Direct answer formatting
    print("\nDirect answer formatting:")
    answer = ResponseTierRouter.format_direct_answer(
        {"data": [{"count": 320}]},
        "user_generic_count",
        "How many employees do we have?"
    )
    print(f"  'How many employees?' -> '{answer}'")


def main():
    print("=" * 60)
    print("  LuBot NVIDIA Routing - Quick Start Demo")
    print("  Extracted from LuBot (112,000+ lines)")
    print("  Live at: https://lubot.ai")
    print("=" * 60)

    if not check_api_key():
        # Still run demos that don't need API key
        print("\nRunning offline demos (no API key needed)...\n")
        demo_intent_classification()
        demo_response_tier()
        return

    # Run all demos
    demo_nvidia_client()
    demo_llm_router()
    demo_embeddings()
    demo_intent_classification()
    demo_response_tier()

    print("\n" + "=" * 60)
    print("  All demos complete!")
    print("  See the full system live at: https://lubot.ai")
    print("=" * 60)


if __name__ == "__main__":
    main()
