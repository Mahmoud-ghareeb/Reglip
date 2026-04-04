"""
Backward-compatible re-exports.

New code should import from ``reglip.embeddings`` instead.
"""

from .embeddings.base import BaseEmbeddingModel
from .embeddings.qwen_api import QwenAPIEmbedding
from .embeddings.omni_embed import OmniEmbedEmbedding
from .embeddings import create_embedding_model, EMBEDDING_REGISTRY

# Legacy alias so existing code that does ``from reglip.embedding_utils import QwenEmbeddingClient``
# keeps working.
QwenEmbeddingClient = QwenAPIEmbedding


def compute_similarity_matrix(embeddings1, embeddings2=None, normalize=True, similarity_metric="dot_product"):
    """Compute similarity matrix between embeddings."""
    import numpy as np
    import torch

    if isinstance(embeddings1, torch.Tensor):
        embeddings1 = embeddings1.cpu().numpy()
    if embeddings2 is not None and isinstance(embeddings2, torch.Tensor):
        embeddings2 = embeddings2.cpu().numpy()
    if embeddings2 is None:
        embeddings2 = embeddings1

    if similarity_metric == "dot_product":
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
    elif similarity_metric == "cosine":
        e1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        e2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        similarity_matrix = np.dot(e1, e2.T)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    if normalize and similarity_metric == "dot_product":
        mn, mx = similarity_matrix.min(), similarity_matrix.max()
        if mx > mn:
            similarity_matrix = (similarity_matrix - mn) / (mx - mn)
        else:
            similarity_matrix = np.zeros_like(similarity_matrix)

    return similarity_matrix


def get_most_similar_labels(query_embeddings, label_embeddings, labels, top_k=1, similarity_metric="dot_product", normalize=True):
    """Find the most similar labels for query embeddings."""
    import numpy as np

    sim = compute_similarity_matrix(query_embeddings, label_embeddings, normalize, similarity_metric)
    results = []
    for row in sim:
        top_idx = np.argsort(row)[-top_k:][::-1]
        results.append([(labels[j], float(row[j])) for j in top_idx])
    return results


def batch_embedding_pipeline(texts, labels, client=None, batch_size=32, similarity_metric="dot_product", normalize=True, top_k=1):
    """Complete pipeline for batch embedding and label prediction."""
    if client is None:
        client = QwenEmbeddingClient()

    text_embeddings = client.get_embeddings(texts, batch_size=batch_size)
    label_embeddings = client.get_embeddings(labels, batch_size=batch_size)
    predictions = get_most_similar_labels(text_embeddings, label_embeddings, labels, top_k=top_k, similarity_metric=similarity_metric, normalize=normalize)
    return text_embeddings, label_embeddings, predictions
