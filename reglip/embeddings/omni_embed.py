"""Tevatron/OmniEmbed-v0.1 — multimodal embedding model served via OpenAI-compatible API."""

import base64
import time
from typing import List

import numpy as np
import requests
from tqdm import tqdm

from .base import BaseEmbeddingModel


class OmniEmbedEmbedding(BaseEmbeddingModel):
    """
    API client for OmniEmbed served behind a vLLM / OpenAI-compatible endpoint.

    Supports both text and image embeddings (cross-modal).
    """

    def __init__(
        self,
        base_url: str = "http://212.41.29.82:6010",
        model: str = "Tevatron/OmniEmbed-v0.1",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/v1/embeddings"
        self._embedding_dim = 3584  # Qwen2.5-Omni hidden size

    # -- BaseEmbeddingModel interface ------------------------------------------

    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> np.ndarray:
        all_embeddings: list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="OmniEmbed text"):
            batch = texts[i : i + batch_size]
            payload = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }
            embs = self._request_with_retry(payload, max_retries, retry_delay)
            all_embeddings.extend(embs)
        return np.array(all_embeddings)

    def get_image_embeddings(
        self,
        images: List[bytes],
        batch_size: int = 8,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> np.ndarray:
        all_embeddings: list = []
        for i in tqdm(range(0, len(images), batch_size), desc="OmniEmbed image"):
            batch = images[i : i + batch_size]
            # Each image becomes a multimodal input with base64-encoded data
            inputs = []
            for img_bytes in batch:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                inputs.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            payload = {
                "model": self.model,
                "input": inputs,
                "encoding_format": "float",
            }
            embs = self._request_with_retry(payload, max_retries, retry_delay)
            all_embeddings.extend(embs)
        return np.array(all_embeddings)

    @property
    def supports_images(self) -> bool:
        return True

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return f"OmniEmbed({self.model})"

    # -- internals -------------------------------------------------------------

    def _request_with_retry(
        self, payload: dict, max_retries: int, retry_delay: float
    ) -> List[List[float]]:
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        delay = retry_delay
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint, json=payload, headers=headers, timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data.get("data", [])]
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(
                        f"Failed to get embeddings after {max_retries + 1} attempts: {e}"
                    )
