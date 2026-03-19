"""Metric computation utilities for evaluation."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics from a similarity matrix.
    
    Args:
        similarity_matrix: (N_images, N_texts) similarity scores
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with i2t and t2i metrics
    """
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
    
    n_images, n_texts = similarity_matrix.shape
    
    # Image-to-Text retrieval (each image has ground truth text at same index)
    i2t_ranks = []
    for i in range(n_images):
        # Get similarities for this image
        sims = similarity_matrix[i]
        # Rank (0-indexed position of correct text when sorted by similarity descending)
        sorted_indices = np.argsort(sims)[::-1]
        rank = np.where(sorted_indices == i)[0][0]
        i2t_ranks.append(rank)
    
    # Text-to-Image retrieval
    t2i_ranks = []
    for i in range(n_texts):
        # Get similarities for this text
        sims = similarity_matrix[:, i]
        # Rank
        sorted_indices = np.argsort(sims)[::-1]
        rank = np.where(sorted_indices == i)[0][0]
        t2i_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)
    
    # Compute Recall@K
    metrics = {}
    for k in k_values:
        metrics[f"i2t_r{k}"] = float(np.mean(i2t_ranks < k) * 100)
        metrics[f"t2i_r{k}"] = float(np.mean(t2i_ranks < k) * 100)
    
    # Mean recall
    all_recalls = [metrics[f"i2t_r{k}"] for k in k_values] + [metrics[f"t2i_r{k}"] for k in k_values]
    metrics["mean_recall"] = float(np.mean(all_recalls))
    
    # Median rank
    metrics["i2t_median_rank"] = float(np.median(i2t_ranks) + 1)  # 1-indexed
    metrics["t2i_median_rank"] = float(np.median(t2i_ranks) + 1)
    
    return metrics


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5],
    multi_labels: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: (N, num_classes) logits or probabilities
        labels: (N,) ground truth labels
        k_values: List of K values for top-K accuracy
        multi_labels: Optional list of lists containing multiple valid labels per sample
                     (for datasets like ImageNet-ReaL)
        
    Returns:
        Dictionary with accuracy metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu()
        labels = labels.cpu()
    
    n_samples = len(labels)
    metrics = {}
    
    for k in k_values:
        # Get top-k predictions
        _, top_k_preds = predictions.topk(k, dim=1)
        
        # Check if correct label is in top-k
        correct = 0
        for i in range(n_samples):
            if multi_labels is not None:
                # Check if any of the valid labels is in top-k
                valid_labels = set(multi_labels[i])
                if any(pred.item() in valid_labels for pred in top_k_preds[i]):
                    correct += 1
            else:
                if labels[i] in top_k_preds[i]:
                    correct += 1
        
        metrics[f"top{k}_accuracy"] = float(correct / n_samples * 100)
    
    return metrics


def compute_real_accuracy(
    predictions: torch.Tensor,
    real_labels: List[List[int]],
    k_values: List[int] = [1, 5],
) -> Dict[str, float]:
    """
    Compute accuracy using ReaL (Re-Assessed Labels) for ImageNet.
    
    ReaL labels allow multiple correct answers per image, accounting for
    labeling ambiguity in the original ImageNet labels.
    
    Args:
        predictions: (N, num_classes) logits or probabilities
        real_labels: List of lists, each containing valid label indices for that image
        k_values: List of K values for top-K accuracy
        
    Returns:
        Dictionary with accuracy metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu()
    
    n_samples = len(real_labels)
    metrics = {}
    
    for k in k_values:
        _, top_k_preds = predictions.topk(k, dim=1)
        
        correct = 0
        for i in range(n_samples):
            valid_labels = set(real_labels[i])
            # Image is correct if any prediction matches any valid label
            if any(pred.item() in valid_labels for pred in top_k_preds[i]):
                correct += 1
        
        metrics[f"real_top{k}_accuracy"] = float(correct / n_samples * 100)
    
    return metrics


def compute_similarity_statistics(
    similarity_matrix: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute statistics about the similarity matrix.
    
    Args:
        similarity_matrix: (N, N) similarity scores
        
    Returns:
        Dictionary with similarity statistics
    """
    if isinstance(similarity_matrix, torch.Tensor):
        sim = similarity_matrix.cpu().numpy()
    else:
        sim = similarity_matrix
    
    n = sim.shape[0]
    
    # Diagonal (matching pairs)
    diagonal = np.diag(sim)
    
    # Off-diagonal (non-matching pairs)
    mask = ~np.eye(n, dtype=bool)
    off_diagonal = sim[mask]
    
    return {
        "diagonal_mean": float(np.mean(diagonal)),
        "diagonal_std": float(np.std(diagonal)),
        "diagonal_min": float(np.min(diagonal)),
        "diagonal_max": float(np.max(diagonal)),
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "off_diagonal_std": float(np.std(off_diagonal)),
        "off_diagonal_min": float(np.min(off_diagonal)),
        "off_diagonal_max": float(np.max(off_diagonal)),
        "gap": float(np.mean(diagonal) - np.mean(off_diagonal)),
    }
