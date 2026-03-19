"""
RegLIP: Enhancing Vision-Language Embeddings through Regression-Based Contrastive Learning

A regression-based approach to vision-language learning that uses continuous similarity scores
instead of binary classification for contrastive learning.
"""

from .config import RegLIPConfig, RegLIPTextConfig, RegLIPVisionConfig
from .model import RegLIPModel, RegLIPOutput
from .utils import load_siglip_checkpoint
from .embedding_utils import (
    QwenEmbeddingClient,
    compute_similarity_matrix,
    get_most_similar_labels,
    batch_embedding_pipeline
)

__version__ = "0.1.0"
__all__ = [
    "RegLIPConfig",
    "RegLIPTextConfig", 
    "RegLIPVisionConfig",
    "RegLIPModel",
    "RegLIPOutput",
    "load_siglip_checkpoint",
    "QwenEmbeddingClient",
    "compute_similarity_matrix",
    "get_most_similar_labels",
    "batch_embedding_pipeline",
] 