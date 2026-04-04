"""Tevatron/OmniEmbed-v0.1 — multimodal embedding model served via custom API."""

import base64
import time
from typing import List

import numpy as np
import requests
from tqdm import tqdm

from .base import BaseEmbeddingModel


class OmniEmbedEmbedding(BaseEmbeddingModel):
    """
    API client for OmniEmbed.

    Text  → POST /v1/embeddings          (input: list[str])
    Image → POST /v1/embeddings/multimodal (input: list of chat-format messages)
    """

    def __init__(
        self,
        base_url: str = "http://212.41.29.82:6010",
        model: str = "Tevatron/OmniEmbed-v0.1",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._text_endpoint = f"{self.base_url}/v1/embeddings"
        self._multimodal_endpoint = f"{self.base_url}/v1/embeddings/multimodal"
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
            embs = self._request_with_retry(
                self._text_endpoint, payload, max_retries, retry_delay
            )
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
            # Build chat-format messages expected by /v1/embeddings/multimodal
            messages_batch = []
            for img_bytes in batch:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                messages_batch.append([{
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{b64}",
                    }],
                }])
            payload = {
                "model": self.model,
                "input": messages_batch,
                "encoding_format": "float",
            }
            embs = self._request_with_retry(
                self._multimodal_endpoint, payload, max_retries, retry_delay
            )
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
        self, endpoint: str, payload: dict, max_retries: int, retry_delay: float
    ) -> List[List[float]]:
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        delay = retry_delay
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    endpoint, json=payload, headers=headers, timeout=120
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
