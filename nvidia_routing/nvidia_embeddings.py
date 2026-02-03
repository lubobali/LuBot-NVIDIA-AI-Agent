"""
NVIDIA Embeddings Client

100% NVIDIA-powered embeddings using nv-embedqa-e5-v5 model.
Replaces MiniLM-L6-v2 (Microsoft) with NVIDIA's embedding model.

Model: nv-embedqa-e5-v5
- Parameters: ~335M
- Dimensions: 1024
- Max tokens: 512
- Optimized for: Q&A retrieval, intent classification

Design Pattern: Same interface as SentenceTransformer for drop-in replacement.
"""

import os
import logging
import numpy as np
from typing import List, Union, Optional

logger = logging.getLogger(__name__)


class NVIDIAEmbeddings:
    """
    NVIDIA Embedding client with SentenceTransformer-compatible interface.

    Drop-in replacement for:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)

    Now:
        from nvidia_routing.nvidia_embeddings import NVIDIAEmbeddings
        model = NVIDIAEmbeddings()
        embeddings = model.encode(texts)
    """

    # NVIDIA embedding model
    MODEL_NAME = "nvidia/nv-embedqa-e5-v5"
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    EMBEDDING_DIM = 1024
    MAX_TOKENS = 512

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize NVIDIA Embeddings client.

        Args:
            api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
            model_name: Override model name (default: nv-embedqa-e5-v5)
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model_name = model_name or self.MODEL_NAME
        self._client = None

        if not self.api_key:
            logger.warning("NVIDIA_API_KEY not set - embeddings will fail")
        else:
            logger.info(f"NVIDIA Embeddings initialized ({self.model_name})")

    def _get_client(self):
        """Lazy-load OpenAI client for NVIDIA API."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.BASE_URL,
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError("OpenAI library required: pip install openai")
        return self._client

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        input_type: str = "query"
    ) -> np.ndarray:
        """
        Encode sentences into embeddings (SentenceTransformer-compatible interface).

        Args:
            sentences: Single string or list of strings to encode
            batch_size: Batch size for API calls (NVIDIA handles batching)
            show_progress_bar: Ignored (for compatibility)
            convert_to_numpy: Return as numpy array (default True)
            normalize_embeddings: L2 normalize embeddings
            input_type: "query" for user questions, "passage" for documents (NVIDIA requirement)

        Returns:
            numpy array of shape (n_sentences, 1024)
        """
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [sentences]

        if not sentences:
            return np.array([])

        try:
            client = self._get_client()

            # Process in batches
            all_embeddings = []

            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]

                # NVIDIA requires input_type for asymmetric models
                response = client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": input_type}
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            embeddings = np.array(all_embeddings)

            # Normalize if requested
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-10)

            logger.debug(f"NVIDIA embeddings: {len(sentences)} texts -> {embeddings.shape}")

            if convert_to_numpy:
                return embeddings
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"NVIDIA embedding failed: {e}")
            # Return zero embeddings as fallback (same shape)
            return np.zeros((len(sentences), self.EMBEDDING_DIM))

    def get_sentence_embedding_dimension(self) -> int:
        """Return embedding dimension (for compatibility)."""
        return self.EMBEDDING_DIM


# =============================================================================
# Singleton instance
# =============================================================================

_nvidia_embeddings: Optional[NVIDIAEmbeddings] = None


def get_nvidia_embeddings() -> NVIDIAEmbeddings:
    """Get singleton NVIDIA embeddings instance."""
    global _nvidia_embeddings
    if _nvidia_embeddings is None:
        _nvidia_embeddings = NVIDIAEmbeddings()
    return _nvidia_embeddings


# =============================================================================
# Drop-in replacement function
# =============================================================================

def encode_texts(texts: Union[str, List[str]], normalize: bool = False) -> np.ndarray:
    """
    Convenience function to encode texts with NVIDIA embeddings.

    Args:
        texts: Single string or list of strings
        normalize: L2 normalize embeddings

    Returns:
        numpy array of embeddings
    """
    return get_nvidia_embeddings().encode(texts, normalize_embeddings=normalize)


# =============================================================================
# Compatibility layer for SentenceTransformer imports
# =============================================================================

def get_embedding_model(model_name: str = None) -> NVIDIAEmbeddings:
    """
    Factory function that returns NVIDIA embeddings regardless of model_name.

    This allows existing code like:
        model = get_embedding_model('all-MiniLM-L6-v2')

    To transparently use NVIDIA instead.
    """
    if model_name and 'minilm' in model_name.lower():
        logger.info(f"Redirecting {model_name} to NVIDIA nv-embedqa-e5-v5")
    return get_nvidia_embeddings()
