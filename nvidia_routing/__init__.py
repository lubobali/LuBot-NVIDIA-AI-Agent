"""
NVIDIA Routing - Multi-model LLM routing for NVIDIA Nemotron

Core components:
- LLMRouter: Unified router with NVIDIA primary + Groq fallback
- NVIDIAClient: Direct NVIDIA Nemotron API wrapper
- NVIDIAEmbeddings: NVIDIA embedding model (nv-embedqa-e5-v5)
- LLMRouterClient: AdalFlow framework adapter
- ResponseTierRouter: 3-tier response routing (direct/enhanced/PhD)
"""

from nvidia_routing.llm_router import LLMRouter, LLMResponse, get_llm_router, chat_tier1, chat_tier2
from nvidia_routing.nvidia_client import NVIDIAClient, NVIDIAResponse, get_nvidia_client
from nvidia_routing.nvidia_embeddings import NVIDIAEmbeddings, get_nvidia_embeddings, encode_texts
from nvidia_routing.response_tier_router import ResponseTierRouter

__all__ = [
    "LLMRouter",
    "LLMResponse",
    "get_llm_router",
    "chat_tier1",
    "chat_tier2",
    "NVIDIAClient",
    "NVIDIAResponse",
    "get_nvidia_client",
    "NVIDIAEmbeddings",
    "get_nvidia_embeddings",
    "encode_texts",
    "ResponseTierRouter",
]
