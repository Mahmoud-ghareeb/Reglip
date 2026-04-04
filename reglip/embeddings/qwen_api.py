"""Qwen embedding model accessed via a remote vLLM / OpenAI-compatible API."""

import time
from typing import List

import numpy as np
import requests
from tqdm import tqdm

from .base import BaseEmbeddingModel


class QwenAPIEmbedding(BaseEmbeddingModel):
    """Client for Qwen3-Embedding-8B served behind an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://212.41.29.82:6010",
        model: str = "Qwen/Qwen3-Embedding-8B",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/v1/embeddings"
        self._embedding_dim = 4096  # Qwen3-Embedding-8B

    # -- BaseEmbeddingModel interface ------------------------------------------

    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> np.ndarray:
        all_embeddings: list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Qwen embeddings"):
            batch = texts[i : i + batch_size]
            payload = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
                "truncate_prompt_tokens": -1,
                "add_special_tokens": True,
                "priority": 0,
            }
            embs = self._request_with_retry(payload, max_retries, retry_delay)
            all_embeddings.extend(embs)
        return np.array(all_embeddings)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return f"QwenAPI({self.model})"

    # -- internals -------------------------------------------------------------

    def _request_with_retry(
        self, payload: dict, max_retries: int, retry_delay: float
    ) -> List[List[float]]:
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        delay = retry_delay
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint, json=payload, headers=headers, timeout=30
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
