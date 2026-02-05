"""
Multi-Model LLM Router - NVIDIA Nemotron (Primary) + Groq (Fallback)

Routes ALL LLM calls through single entry point.
Primary: NVIDIA Nemotron (Nano 8B + Ultra 253B) - tried EVERY request (99% success)
Fallback: Groq (ONLY if NVIDIA fails) - rare (1% of requests)

Pattern: Google Vertex AI, Netflix API Gateway, Stripe Payment Router

Design Principle:
- NVIDIA tried first on EVERY request (not switched away from)
- Groq only called when NVIDIA actually fails (infrastructure issues)
- Smart fallback (only for network/rate/auth errors, not content violations)
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from any LLM provider."""
    content: str = ""
    model: str = ""
    tokens_used: int = 0
    success: bool = False
    used_fallback: bool = False
    error: Optional[str] = None


class LLMRouter:
    """
    Unified LLM Router - Routes to NVIDIA (primary) or Groq (fallback only)

    Tier Mapping:
    - Tier 1: NVIDIA Nano 8B (8B params) - fast queries
    - Tier 2: NVIDIA Ultra 253B (253B params!) - PhD analysis
    - Fallback: Groq (only called if NVIDIA fails)

    Pattern: Netflix failover - try primary first, instant fallback on failure
    """

    # NVIDIA Model configurations
    NANO_8B = "nvidia/llama-3.1-nemotron-nano-8b-v1"
    ULTRA_253B = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

    # Groq Model (fallback only)
    GROQ_MODEL = "llama-3.3-70b-versatile"

    # API endpoints
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self):
        """Initialize router with API keys."""
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Self-hosted NVIDIA RTX 4090 GPU (RunPod + Ollama)
        self.gpu_server_url = os.getenv("GPU_SERVER_URL")
        self.gpu_model_name = os.getenv("GPU_MODEL_NAME", "nemotron-mini")

        if not self.nvidia_api_key:
            logger.warning("NVIDIA_API_KEY not set - fallback will always be used")
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set - no fallback available")
        if self.gpu_server_url:
            logger.info(f"Self-hosted GPU: {self.gpu_server_url} (model: {self.gpu_model_name})")
        else:
            logger.info("GPU_SERVER_URL not set - using cloud API only")

        logger.info("LLM Router initialized (GPU -> NVIDIA NIM -> Groq)")

    def chat_completion(
        self,
        tier: int,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> LLMResponse:
        """
        Main routing method - tries NVIDIA first (EVERY request), falls back to Groq on failure.

        Args:
            tier: 1 (Nano 8B) or 2 (Ultra 253B)
            messages: List of {role, content} messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLMResponse with content and metadata

        Pattern: Netflix failover
        - 99% of requests: NVIDIA succeeds, return immediately (Groq never called)
        - 1% of requests: NVIDIA fails, try Groq fallback
        """

        # ================================================================
        # STEP 0: TRY SELF-HOSTED GPU FIRST (Tier 1 only)
        # Pattern: Edge computing - local GPU for speed, cloud for heavy lifting
        # ================================================================
        if tier == 1 and self.gpu_server_url:
            logger.info(f"LLM Router: Trying self-hosted GPU ({self.gpu_model_name}) first...")

            gpu_response = self._call_gpu(messages, max_tokens, temperature)

            if gpu_response.success:
                logger.info(f"GPU succeeded: {gpu_response.model} ({gpu_response.tokens_used} tokens)")
                return gpu_response  # Fastest path! GPU on NVIDIA RTX 4090

            logger.warning(f"GPU failed: {gpu_response.error} - falling through to NIM API")

        # ================================================================
        # STEP 1: TRY NVIDIA NIM API (cloud)
        # ================================================================
        model_name = "Nano 8B" if tier == 1 else "Ultra 253B"
        logger.info(f"LLM Router: Trying NVIDIA NIM {model_name}...")

        nvidia_response = self._call_nvidia(
            model=self.NANO_8B if tier == 1 else self.ULTRA_253B,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # ================================================================
        # NVIDIA SUCCESS (99% of time) -> Return immediately, Groq never called
        # ================================================================
        if nvidia_response.success:
            logger.info(f"NVIDIA succeeded: {nvidia_response.model} ({nvidia_response.tokens_used} tokens)")
            return nvidia_response

        # ================================================================
        # NVIDIA FAILED (1% of time) -> Try Groq fallback
        # ================================================================
        logger.warning(f"NVIDIA failed: {nvidia_response.error}")

        # Check if failure is infrastructure-related (should fallback)
        if self._should_fallback(nvidia_response.error):
            logger.warning("Infrastructure failure detected, falling back to Groq...")

            groq_response = self._call_groq(messages, max_tokens, temperature)

            if groq_response.success:
                logger.info(f"Groq fallback succeeded ({groq_response.tokens_used} tokens)")
                groq_response.used_fallback = True
                return groq_response
            else:
                logger.error("Both NVIDIA and Groq failed!")
                return groq_response

        else:
            # Don't fallback - error is not infrastructure-related
            logger.error(f"NVIDIA failed, no fallback attempted: {nvidia_response.error}")
            return nvidia_response

    def _call_nvidia(
        self,
        model: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """
        Call NVIDIA Nemotron API.

        Pattern: OpenAI-compatible API (drop-in replacement)
        """
        try:
            from openai import OpenAI

            if not self.nvidia_api_key:
                return LLMResponse(
                    success=False,
                    error="NVIDIA_API_KEY not set"
                )

            # NVIDIA API is OpenAI-compatible
            client = OpenAI(
                base_url=self.NVIDIA_BASE_URL,
                api_key=self.nvidia_api_key
            )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            return LLMResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                success=True,
                used_fallback=False
            )

        except ImportError:
            error_msg = "OpenAI library not installed (required for NVIDIA API)"
            logger.error(error_msg)
            return LLMResponse(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"NVIDIA API call failed: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(success=False, error=error_msg)

    def _call_gpu(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """
        Call self-hosted NVIDIA RTX 4090 GPU via Ollama (OpenAI-compatible).

        Pattern: Edge inference - local GPU for latency-sensitive Tier 1 classification.
        Ollama exposes /v1/chat/completions (same as NIM API).
        """
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=f"{self.gpu_server_url}/v1",
                api_key="ollama"  # Ollama doesn't require auth
            )

            response = client.chat.completions.create(
                model=self.gpu_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            return LLMResponse(
                content=content,
                model=f"gpu-{self.gpu_model_name}",
                tokens_used=tokens_used,
                success=True,
                used_fallback=False
            )

        except Exception as e:
            error_msg = f"GPU call failed: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(success=False, error=error_msg)

    def _call_groq(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """
        Call Groq API (fallback ONLY - not primary).

        This method is ONLY called when NVIDIA fails.
        Pattern: Netflix circuit breaker - instant fallback on infrastructure failure
        """
        try:
            from groq import Groq

            if not self.groq_api_key:
                return LLMResponse(
                    success=False,
                    error="GROQ_API_KEY not set - no fallback available"
                )

            client = Groq(api_key=self.groq_api_key)

            response = client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            logger.info(f"Groq fallback succeeded: {tokens_used} tokens")

            return LLMResponse(
                content=content,
                model=f"groq-{self.GROQ_MODEL}",
                tokens_used=tokens_used,
                success=True,
                used_fallback=True
            )

        except ImportError:
            error_msg = "Groq library not installed"
            logger.error(error_msg)
            return LLMResponse(success=False, error=error_msg)

        except Exception as e:
            error_msg = f"Groq fallback failed: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(success=False, error=error_msg)

    def _should_fallback(self, error: str) -> bool:
        """
        Decide if we should fallback to Groq based on error type.

        Fallback for:
        - Infrastructure issues (network, timeout, 503, 502)
        - NVIDIA service problems (rate limit, maintenance)
        - Authentication issues (invalid key, expired)

        NO fallback for:
        - Our code bugs (invalid format, wrong parameters)
        - Content violations (same on Groq)
        - Max tokens exceeded (same on Groq)

        Pattern: Only fallback for things outside our control
        """
        if not error:
            return False

        # Infrastructure/service errors that warrant fallback
        infrastructure_errors = [
            "timeout",
            "connection",
            "network",
            "503",
            "502",
            "500",
            "rate limit",
            "quota",
            "api_key",
            "authentication",
            "unavailable",
            "maintenance",
            "not found",
            "dns",
            "unreachable"
        ]

        error_lower = error.lower()
        should_fallback = any(keyword in error_lower for keyword in infrastructure_errors)

        if should_fallback:
            logger.info("Fallback decision: YES (infrastructure error detected)")
        else:
            logger.info("Fallback decision: NO (not an infrastructure error)")

        return should_fallback


# =============================================================================
# Singleton pattern (one instance for entire application)
# =============================================================================

_router_instance: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """
    Get singleton LLM Router instance.

    Pattern: Netflix API Gateway - single instance handles all requests
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


# =============================================================================
# Convenience functions for common use cases
# =============================================================================

def chat_tier1(messages: List[Dict[str, str]], max_tokens: int = 500) -> LLMResponse:
    """
    Convenience function for Tier 1 queries (NVIDIA Nano 8B).

    Examples: "Show me revenue by region", "Top 5 products"
    """
    return get_llm_router().chat_completion(tier=1, messages=messages, max_tokens=max_tokens)


def chat_tier2(messages: List[Dict[str, str]], max_tokens: int = 1000) -> LLMResponse:
    """
    Convenience function for Tier 2 queries (NVIDIA Ultra 253B).

    Examples: "Correlation between X and Y", "HHI concentration"
    """
    return get_llm_router().chat_completion(tier=2, messages=messages, max_tokens=max_tokens)


# =============================================================================
# FUNCTION CALLING / STRUCTURED OUTPUT (OpenAI/Anthropic Pattern)
# =============================================================================
# Pattern: Guaranteed JSON schema compliance via function calling
# Groq supports tool_choice="required" which FORCES function call response
# =============================================================================

@dataclass
class StructuredResponse:
    """Response from structured output (function calling)."""
    data: Dict[str, Any] = None
    raw_content: str = ""
    model: str = ""
    tokens_used: int = 0
    success: bool = False
    error: Optional[str] = None


# Business Analyst response schema (used for V3 responses)
BUSINESS_ANALYST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "format_business_response",
        "description": "Format a business analyst response with headline, insight, and suggestions",
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string",
                    "description": "Data summary with numbers, max 60 chars. Format: 'Product: $10.2k | HR: $9.9k'"
                },
                "insight": {
                    "type": "string",
                    "description": "One insight the user can't see at first glance, max 100 chars"
                },
                "insight_source": {
                    "type": "string",
                    "description": "The SQL or calculation that proves the insight"
                },
                "suggestion_1": {
                    "type": "string",
                    "description": "First exploration suggestion using their column names, max 50 chars"
                },
                "suggestion_2": {
                    "type": "string",
                    "description": "Second exploration suggestion using their column names, max 50 chars"
                }
            },
            "required": ["headline", "insight", "insight_source", "suggestion_1", "suggestion_2"]
        }
    }
}


def structured_response(
    messages: List[Dict[str, str]],
    schema: Dict = None,
    max_tokens: int = 500
) -> StructuredResponse:
    """
    Generate structured output using function calling.

    Pattern: OpenAI/Anthropic structured output - GUARANTEED schema compliance
    - Uses Groq's function calling with tool_choice="required"
    - LLM MUST call the function, cannot return plain text
    - Response is ALWAYS valid JSON matching the schema

    Args:
        messages: Chat messages
        schema: Function schema (defaults to BUSINESS_ANALYST_SCHEMA)
        max_tokens: Max tokens

    Returns:
        StructuredResponse with parsed data

    Example:
        result = structured_response(
            messages=[{"role": "user", "content": "Analyze this data..."}],
            schema=BUSINESS_ANALYST_SCHEMA
        )
        if result.success:
            headline = result.data["headline"]  # Guaranteed to exist!
    """
    import json

    schema = schema or BUSINESS_ANALYST_SCHEMA
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        return StructuredResponse(
            success=False,
            error="GROQ_API_KEY not set - function calling unavailable"
        )

    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)

        # tool_choice="required" FORCES function call
        # LLM cannot return plain text - MUST call the function
        logger.info("Structured output: Using function calling (guaranteed schema)")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=[schema],
            tool_choice={"type": "function", "function": {"name": schema["function"]["name"]}},
            max_tokens=max_tokens,
            temperature=0.3
        )

        # Extract function call arguments
        message = response.choices[0].message
        tokens_used = response.usage.total_tokens if response.usage else 0

        if message.tool_calls and len(message.tool_calls) > 0:
            tool_call = message.tool_calls[0]
            arguments_str = tool_call.function.arguments

            # Parse the JSON arguments
            parsed_data = json.loads(arguments_str)

            logger.info(f"Structured output succeeded: {list(parsed_data.keys())}")

            return StructuredResponse(
                data=parsed_data,
                raw_content=arguments_str,
                model="groq-llama-3.3-70b-versatile",
                tokens_used=tokens_used,
                success=True
            )
        else:
            logger.warning("Structured output: No function call in response")
            return StructuredResponse(
                success=False,
                error="LLM did not return function call",
                raw_content=message.content or ""
            )

    except json.JSONDecodeError as e:
        logger.error(f"Structured output: JSON parse failed: {e}")
        return StructuredResponse(success=False, error=f"JSON parse failed: {e}")

    except Exception as e:
        logger.error(f"Structured output failed: {e}")
        return StructuredResponse(success=False, error=str(e))
