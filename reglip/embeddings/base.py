"""Base class for all embedding models."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models used as frozen teachers in RegLIP."""

    @abstractmethod
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of input texts.
            batch_size: Number of texts to process per batch.

        Returns:
            numpy array of shape (num_texts, embedding_dim).
        """
        ...

    def get_image_embeddings(self, images: List[bytes], batch_size: int = 8) -> np.ndarray:
        """
        Get embeddings for a list of images (raw JPEG/PNG bytes).

        Override this in multimodal embedding models.

        Args:
            images: List of raw image bytes (JPEG/PNG).
            batch_size: Number of images to process per batch.

        Returns:
            numpy array of shape (num_images, embedding_dim).
        """
        raise NotImplementedError(f"{self.name} does not support image embeddings")

    @property
    def supports_images(self) -> bool:
        """Whether this embedding model can embed images."""
        return False

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for logging."""
        ...
