"""
NVIDIA Nemotron Client - Direct API Wrapper

NVIDIA API Integration for LuBot's 3-Tier Response System:
- Tier 0 (Direct): No LLM needed - formatted text
- Tier 1 (Enhanced): NVIDIA Nemotron Nano 8B (simple queries, fast)
- Tier 2 (Full PhD): NVIDIA Nemotron Ultra 253B (PhD-level statistical analysis)

Design Pattern: Multi-model routing (Google uses different models for different tasks)
- Small model for simple tasks (cost-effective, fast)
- Large model for complex tasks (high quality)

API Documentation: https://build.nvidia.com/nvidia/nemotron-4-340b-reward
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NVIDIAResponse:
    """Response from NVIDIA API."""
    content: str
    model: str
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class NVIDIAClient:
    """
    NVIDIA Nemotron API client with multi-model routing.

    Models:
    - Nano 8B: Fast, cost-effective for simple tasks (Tier 1)
    - Ultra 253B: PhD-level accuracy for complex analysis (Tier 2)

    Pattern: OpenAI-compatible API (drop-in replacement)
    """

    # Model configurations
    NANO_8B = "nvidia/llama-3.1-nemotron-nano-8b-v1"   # Tier 1: Fast model (8B params)
    ULTRA_253B = "nvidia/llama-3.1-nemotron-ultra-253b-v1"  # Tier 2: PhD-level (253B params!)

    # API endpoint
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str = None):
        """
        Initialize NVIDIA client.

        Args:
            api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY not set - NVIDIA calls will fail")

        logger.info("NVIDIA Client initialized (Nemotron models)")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = NANO_8B,
        max_tokens: int = 500,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> NVIDIAResponse:
        """
        Call NVIDIA chat completion API.

        Args:
            messages: List of {role, content} messages
            model: Model name (NANO_8B or ULTRA_253B)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling threshold

        Returns:
            NVIDIAResponse with content and metadata
        """
        try:
            from openai import OpenAI

            # NVIDIA API is OpenAI-compatible
            client = OpenAI(
                base_url=self.BASE_URL,
                api_key=self.api_key
            )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            logger.info(f"NVIDIA {model}: {tokens_used} tokens")

            return NVIDIAResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                success=True
            )

        except ImportError as e:
            error_msg = "OpenAI library not installed (required for NVIDIA API)"
            logger.error(error_msg)
            return NVIDIAResponse(
                content="",
                model=model,
                success=False,
                error=error_msg
            )

        except Exception as e:
            error_msg = f"NVIDIA API call failed: {str(e)}"
            logger.error(error_msg)
            return NVIDIAResponse(
                content="",
                model=model,
                success=False,
                error=error_msg
            )

    def chat_tier1(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500
    ) -> NVIDIAResponse:
        """
        Tier 1 (Enhanced) queries - use fast Nano 8B model.

        Examples:
        - "Show me revenue by region"
        - "What are the top 5 products?"
        - "Compare sales between departments"
        """
        logger.info("NVIDIA Tier 1 (Nano 8B): Simple analysis")
        return self.chat_completion(
            messages=messages,
            model=self.NANO_8B,
            max_tokens=max_tokens,
            temperature=0.3
        )

    def chat_tier2(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000
    ) -> NVIDIAResponse:
        """
        Tier 2 (Full PhD) queries - use Ultra 253B model.

        Examples:
        - "What's the correlation between X and Y?"
        - "Calculate HHI for market concentration"
        - "Run t-test to compare segments"
        """
        logger.info("NVIDIA Tier 2 (Ultra 253B): PhD analysis")
        return self.chat_completion(
            messages=messages,
            model=self.ULTRA_253B,
            max_tokens=max_tokens,
            temperature=0.2
        )


# =============================================================================
# Singleton instance
# =============================================================================

_nvidia_client: Optional[NVIDIAClient] = None


def get_nvidia_client() -> NVIDIAClient:
    """Get singleton NVIDIA client instance."""
    global _nvidia_client
    if _nvidia_client is None:
        _nvidia_client = NVIDIAClient()
    return _nvidia_client


# =============================================================================
# Convenience functions
# =============================================================================

def nvidia_chat_tier1(messages: List[Dict[str, str]], max_tokens: int = 500) -> NVIDIAResponse:
    """Convenience function for Tier 1 queries."""
    return get_nvidia_client().chat_tier1(messages, max_tokens)


def nvidia_chat_tier2(messages: List[Dict[str, str]], max_tokens: int = 1000) -> NVIDIAResponse:
    """Convenience function for Tier 2 queries."""
    return get_nvidia_client().chat_tier2(messages, max_tokens)
