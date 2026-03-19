"""
Utility functions for batch text embedding using external APIs and computing similarities.
"""

import requests
import numpy as np
import torch
from typing import List, Union, Tuple, Optional
from tqdm import tqdm
import time


class QwenEmbeddingClient:
    """Client for interacting with Qwen embedding API."""
    
    def __init__(self, base_url: str = "http://212.41.29.82:6010", model: str = "Qwen/Qwen3-Embedding-8B"):
        """
        Initialize the embedding client.
        
        Args:
            base_url: Base URL of the embedding API
            model: Model name to use for embeddings
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.endpoint = f"{self.base_url}/v1/embeddings"
        
    def get_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        encoding_format: str = "float",
        add_special_tokens: bool = True,
        truncate_prompt_tokens: int = -1,
        priority: int = 0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> np.ndarray:
        """
        Get embeddings for a list of texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process per API call
            encoding_format: Format for embeddings ('float' or 'base64')
            add_special_tokens: Whether to add special tokens
            truncate_prompt_tokens: Token truncation limit (-1 for no limit)
            priority: Request priority
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            
        Returns:
            numpy array of embeddings with shape (num_texts, embedding_dim)
        """
        all_embeddings = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "input": batch_texts,
                "encoding_format": encoding_format,
                "truncate_prompt_tokens": truncate_prompt_tokens,
                "add_special_tokens": add_special_tokens,
                "priority": priority
            }
            
            # Make API request with retries
            embeddings_batch = self._make_request_with_retry(
                payload, max_retries, retry_delay
            )
            all_embeddings.extend(embeddings_batch)
            
        return np.array(all_embeddings)
    
    def _make_request_with_retry(
        self, 
        payload: dict, 
        max_retries: int, 
        retry_delay: float
    ) -> List[List[float]]:
        """Make API request with retry logic."""
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                
                # Extract embeddings from response
                data = response.json()
                embeddings = []
                for item in data.get('data', []):
                    embeddings.append(item['embedding'])
                
                return embeddings
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"Failed to get embeddings after {max_retries + 1} attempts: {e}")


def compute_similarity_matrix(
    embeddings1: Union[np.ndarray, torch.Tensor],
    embeddings2: Optional[Union[np.ndarray, torch.Tensor]] = None,
    normalize: bool = True,
    similarity_metric: str = "dot_product"
) -> np.ndarray:
    """
    Compute similarity matrix between embeddings using dot product.
    
    Args:
        embeddings1: First set of embeddings (N x D)
        embeddings2: Second set of embeddings (M x D). If None, compute self-similarity
        normalize: Whether to normalize results between 0 and 1
        similarity_metric: Similarity metric to use ('dot_product', 'cosine')
        
    Returns:
        Similarity matrix of shape (N x M) or (N x N) if embeddings2 is None
    """
    # Convert to numpy if needed
    if isinstance(embeddings1, torch.Tensor):
        embeddings1 = embeddings1.cpu().numpy()
    if embeddings2 is not None and isinstance(embeddings2, torch.Tensor):
        embeddings2 = embeddings2.cpu().numpy()
    
    # Use embeddings1 for both if embeddings2 is not provided
    if embeddings2 is None:
        embeddings2 = embeddings1
    
    if similarity_metric == "dot_product":
        # Compute dot product similarity
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
    elif similarity_metric == "cosine":
        # Normalize embeddings for cosine similarity
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
        
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
    
    if normalize and similarity_metric == "dot_product":
        # Normalize to [0, 1] range
        min_val = similarity_matrix.min()
        max_val = similarity_matrix.max()
        if max_val > min_val:
            similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)
        else:
            similarity_matrix = np.zeros_like(similarity_matrix)
    
    return similarity_matrix


def get_most_similar_labels(
    query_embeddings: Union[np.ndarray, torch.Tensor],
    label_embeddings: Union[np.ndarray, torch.Tensor],
    labels: List[str],
    top_k: int = 1,
    similarity_metric: str = "dot_product",
    normalize: bool = True
) -> List[Tuple[str, float]]:
    """
    Find the most similar labels for query embeddings.
    
    Args:
        query_embeddings: Query embeddings (N x D)
        label_embeddings: Label embeddings (M x D)
        labels: List of label names corresponding to label_embeddings
        top_k: Number of top similar labels to return
        similarity_metric: Similarity metric to use
        normalize: Whether to normalize similarity scores
        
    Returns:
        List of tuples (label, similarity_score) for each query
    """
    similarity_matrix = compute_similarity_matrix(
        query_embeddings, label_embeddings, normalize, similarity_metric
    )
    
    results = []
    for i, query_similarities in enumerate(similarity_matrix):
        # Get top-k most similar labels
        top_indices = np.argsort(query_similarities)[-top_k:][::-1]
        query_results = []
        for idx in top_indices:
            query_results.append((labels[idx], float(query_similarities[idx])))
        results.append(query_results)
    
    return results


def batch_embedding_pipeline(
    texts: List[str],
    labels: List[str],
    client: Optional[QwenEmbeddingClient] = None,
    batch_size: int = 32,
    similarity_metric: str = "dot_product",
    normalize: bool = True,
    top_k: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, float]]]:
    """
    Complete pipeline for batch embedding and label prediction.
    
    Args:
        texts: List of input texts to embed
        labels: List of possible labels
        client: QwenEmbeddingClient instance (creates default if None)
        batch_size: Batch size for API calls
        similarity_metric: Similarity metric to use
        normalize: Whether to normalize similarity scores
        top_k: Number of top labels to return for each text
        
    Returns:
        Tuple of (text_embeddings, label_embeddings, predictions)
    """
    if client is None:
        client = QwenEmbeddingClient()
    
    print("Getting embeddings for texts...")
    text_embeddings = client.get_embeddings(texts, batch_size=batch_size)
    
    print("Getting embeddings for labels...")
    label_embeddings = client.get_embeddings(labels, batch_size=batch_size)
    
    print("Computing similarities and predictions...")
    predictions = get_most_similar_labels(
        text_embeddings, label_embeddings, labels, 
        top_k=top_k, similarity_metric=similarity_metric, normalize=normalize
    )
    
    return text_embeddings, label_embeddings, predictions


# Example usage
if __name__ == "__main__":
    # Example usage
    client = QwenEmbeddingClient()
    
    # Sample texts and labels
    texts = [
        "hello my name is mahmoud",
        "I love machine learning",
        "The weather is nice today"
    ]
    
    labels = [
        "greeting",
        "technology",
        "weather",
        "personal information"
    ]
    
    # Run the complete pipeline
    text_embeddings, label_embeddings, predictions = batch_embedding_pipeline(
        texts=texts,
        labels=labels,
        client=client,
        batch_size=4,
        similarity_metric="dot_product",
        normalize=True,
        top_k=2
    )
    
    # Print results
    for i, (text, pred_list) in enumerate(zip(texts, predictions)):
        print(f"\nText: '{text}'")
        print("Top predictions:")
        for label, score in pred_list:
            print(f"  {label}: {score:.4f}") 