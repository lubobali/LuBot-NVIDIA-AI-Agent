"""
AdalFlow-Compatible LLM Router Client

PURPOSE:
AdalFlow Generator wrapper that routes through Unified LLM Router.
Makes AdalFlow use NVIDIA (primary) + Groq (fallback) transparently.

PATTERN:
Adapter Pattern - wraps our router to match AdalFlow's client interface

USAGE:
    from nvidia_routing.llm_router_client import LLMRouterClient

    # Drop-in replacement for GroqAPIClient
    model_client = LLMRouterClient(tier=1)  # Tier 1 = Nano 8B, Tier 2 = Ultra 253B

    generator = Generator(
        model_client=model_client,
        model_kwargs={"temperature": 0.3, "max_tokens": 500}
    )
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from adalflow.core.types import ModelType, GeneratorOutput, CompletionUsage as AdalflowUsage
from adalflow.core.model_client import ModelClient

from nvidia_routing.llm_router import get_llm_router

# =============================================================================
# OpenAI Types Import (NVIDIA NIM API is 100% OpenAI-compatible)
# =============================================================================
try:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types import CompletionUsage
    OPENAI_TYPES_AVAILABLE = True
except ImportError:
    OPENAI_TYPES_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# AdalFlow-Compatible Response Objects (Adapter Pattern)
# =============================================================================

@dataclass
class MockAPIResponse:
    """
    Mock API response for AdalFlow compatibility.

    Mimics OpenAI/Groq ChatCompletion structure.
    Forward compatible with future API changes.
    """
    content: str
    model: str
    usage: Dict = field(default_factory=dict)

    # Additional fields for full compatibility
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    choices: List = field(default_factory=list)


@dataclass
class AdalflowResponse:
    """
    AdalFlow-compatible response object.

    Pattern: Adapter Pattern
    - Mimics GroqAPIClient/OpenAIClient response structure
    - Compatible with AdalFlow Generator expectations
    """
    data: str
    model: str = ""
    usage: Dict = None
    api_response: Any = None
    raw_response: str = ""
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


class LLMRouterClient(ModelClient):
    """
    AdalFlow-compatible client that routes through Unified LLM Router.

    Drop-in replacement for GroqAPIClient:
    - Tier 1: NVIDIA Nano 8B (fast, simple queries)
    - Tier 2: NVIDIA Ultra 253B (complex, PhD analysis)
    - Fallback: Groq (only if NVIDIA fails)

    Pattern: Adapter Pattern (wrap router to match AdalFlow interface)
    """

    def __init__(
        self,
        tier: int = 1,  # 1 = Nano 8B, 2 = Ultra 253B
        **kwargs
    ):
        """
        Initialize LLMRouterClient.

        Args:
            tier: Model tier (1 = Nano 8B, 2 = Ultra 253B)
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()
        self.tier = tier
        self.router = get_llm_router()

        # Model info for AdalFlow
        self.model_type = ModelType.LLM

        logger.info(f"LLMRouterClient initialized (Tier {tier})")

    def call(
        self,
        api_kwargs: Dict[str, Any] = None,
        model_kwargs: Dict[str, Any] = None,
        **kwargs
    ) -> Any:
        """
        Main call method - routes through Unified LLM Router.

        Args:
            api_kwargs: API-specific arguments (messages, etc.) OR a string prompt
            model_kwargs: Model arguments (temperature, max_tokens, etc.)

        Returns:
            AdalFlow-compatible response
        """
        # Handle string prompts (flexible API)
        if isinstance(api_kwargs, str):
            messages = [{"role": "user", "content": api_kwargs}]
            api_kwargs = {}
        else:
            api_kwargs = api_kwargs or {}
            messages = api_kwargs.get("messages", [])

        model_kwargs = model_kwargs or {}
        temperature = model_kwargs.get("temperature", 0.3)
        max_tokens = model_kwargs.get("max_tokens", 500 if self.tier == 1 else 1000)

        logger.info(f"LLMRouterClient routing call (Tier {self.tier})...")

        # Route through unified router (NVIDIA primary, Groq fallback)
        llm_response = self.router.chat_completion(
            tier=self.tier,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # =================================================================
        # Return OpenAI ChatCompletion Object
        # =================================================================
        # NVIDIA NIM API is 100% OpenAI-compatible
        # AdalFlow Generator expects: response.choices[0].message.content
        # We build a real ChatCompletion object (same as NVIDIA/Groq returns)
        # =================================================================
        if llm_response.success:
            timestamp = int(time.time())

            if OPENAI_TYPES_AVAILABLE:
                # Build real OpenAI ChatCompletion object
                completion = ChatCompletion(
                    id=f"chatcmpl-nvidia-{timestamp}",
                    choices=[
                        Choice(
                            finish_reason="stop",
                            index=0,
                            message=ChatCompletionMessage(
                                content=llm_response.content,
                                role="assistant"
                            )
                        )
                    ],
                    created=timestamp,
                    model=llm_response.model,
                    object="chat.completion",
                    usage=CompletionUsage(
                        completion_tokens=llm_response.tokens_used or 0,
                        prompt_tokens=0,
                        total_tokens=llm_response.tokens_used or 0
                    )
                )
                logger.info(f"LLMRouterClient returning ChatCompletion (model={llm_response.model})")
                return completion
            else:
                # Fallback: Return dict-like object (legacy compatibility)
                logger.warning("OpenAI types not available, using fallback response")
                return AdalflowResponse(
                    data=llm_response.content,
                    model=llm_response.model,
                    usage={"completion_tokens": llm_response.tokens_used},
                    api_response=MockAPIResponse(
                        content=llm_response.content,
                        model=llm_response.model,
                        usage={"completion_tokens": llm_response.tokens_used},
                        id=f"chatcmpl-{timestamp}",
                        created=timestamp,
                        choices=[{"message": {"content": llm_response.content, "role": "assistant"}}]
                    ),
                    raw_response=llm_response.content,
                    metadata={"tier": self.tier}
                )
        else:
            # Failed - raise error for AdalFlow to handle
            raise RuntimeError(f"LLM call failed: {llm_response.error}")

    def convert_inputs_to_api_kwargs(
        self,
        input: Any = None,
        model_kwargs: Dict = None,
        model_type: ModelType = ModelType.LLM
    ) -> Dict[str, Any]:
        """
        Convert AdalFlow inputs to API kwargs.

        This method is called by AdalFlow Generator to prepare the API call.

        Args:
            input: Input data (usually a string prompt or messages list)
            model_kwargs: Model arguments
            model_type: Type of model (LLM, EMBEDDER, etc.)

        Returns:
            API kwargs dict with "messages" key
        """
        model_kwargs = model_kwargs or {}

        # Convert input to messages format
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = input
        elif isinstance(input, dict):
            messages = input.get("messages", [])
        else:
            messages = [{"role": "user", "content": str(input)}]

        return {
            "messages": messages,
            **model_kwargs
        }

    def parse_chat_completion(self, completion: Any) -> "GeneratorOutput":
        """
        Parse response from LLM call - returns GeneratorOutput like GroqAPIClient.

        Args:
            completion: Response from call() - ChatCompletion object

        Returns:
            GeneratorOutput with data=None, usage=usage, raw_response=content
        """
        try:
            data = completion.choices[0].message.content
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, usage=usage, raw_response=data)

        except Exception as e:
            logger.error(f"Error parsing completion: {e}")
            return GeneratorOutput(data=str(e), usage=None, raw_response=completion)

    def track_completion_usage(self, completion: Any) -> AdalflowUsage:
        """Track token usage from completion - mirrors GroqAPIClient pattern."""
        try:
            if hasattr(completion, 'usage') and completion.usage:
                return AdalflowUsage(
                    completion_tokens=completion.usage.completion_tokens or 0,
                    prompt_tokens=completion.usage.prompt_tokens or 0,
                    total_tokens=completion.usage.total_tokens or 0
                )
        except Exception:
            pass
        return AdalflowUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)


def get_llm_router_client(tier: int = 1) -> LLMRouterClient:
    """
    Get LLMRouterClient instance.

    Args:
        tier: Model tier (1 = Nano 8B, 2 = Ultra 253B)

    Returns:
        LLMRouterClient instance
    """
    return LLMRouterClient(tier=tier)


if __name__ == "__main__":
    # Quick test
    print("Testing LLMRouterClient...")

    client = LLMRouterClient(tier=1)

    # Test call
    api_kwargs = {
        "messages": [{"role": "user", "content": "Say 'Hello from NVIDIA Nano 8B!'"}]
    }
    model_kwargs = {
        "temperature": 0.3,
        "max_tokens": 50
    }

    try:
        response = client.call(api_kwargs=api_kwargs, model_kwargs=model_kwargs)
        print(f"\nSuccess!")
        print(f"Model: {response.model}")
        print(f"Response: {response.data}")
    except Exception as e:
        print(f"\nFailed: {e}")
