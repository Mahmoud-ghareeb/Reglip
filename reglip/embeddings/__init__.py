"""
Embedding models used as frozen teachers in RegLIP.

Usage:
    from reglip.embeddings import create_embedding_model

    model = create_embedding_model("qwen_api", base_url="...", model="...")
    model = create_embedding_model("omni_embed")
"""

from .base import BaseEmbeddingModel
from .qwen_api import QwenAPIEmbedding
from .omni_embed import OmniEmbedEmbedding

# ---------------------------------------------------------------------------
# Registry — add new models here
# ---------------------------------------------------------------------------
EMBEDDING_REGISTRY: dict[str, type[BaseEmbeddingModel]] = {
    "qwen_api": QwenAPIEmbedding,
    "omni_embed": OmniEmbedEmbedding,
}


def create_embedding_model(name: str, **kwargs) -> BaseEmbeddingModel:
    """
    Factory that instantiates an embedding model by its registry key.

    Args:
        name: Key in EMBEDDING_REGISTRY (e.g. "qwen_api", "omni_embed").
        **kwargs: Forwarded to the model constructor.

    Returns:
        An instance of BaseEmbeddingModel.
    """
    if name not in EMBEDDING_REGISTRY:
        available = ", ".join(sorted(EMBEDDING_REGISTRY.keys()))
        raise ValueError(f"Unknown embedding model '{name}'. Available: {available}")

    cls = EMBEDDING_REGISTRY[name]
    return cls(**kwargs)


__all__ = [
    "BaseEmbeddingModel",
    "QwenAPIEmbedding",
    "OmniEmbedEmbedding",
    "EMBEDDING_REGISTRY",
    "create_embedding_model",
]
