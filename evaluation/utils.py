"""Utility functions for evaluation."""

import os
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract_image_features(
    model,
    dataloader,
    device: str = "cuda",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extract image features from a model.
    
    Args:
        model: Vision-language model
        dataloader: DataLoader yielding batches with 'pixel_values'
        device: Device to use
        normalize: Whether to L2-normalize features
        
    Returns:
        Tensor of image features (N, D)
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image features"):
            pixel_values = batch['pixel_values'].to(device)
            
            # Get image features
            image_features = model.get_image_features(pixel_values)
            
            if normalize:
                image_features = F.normalize(image_features, p=2, dim=-1)
            
            all_features.append(image_features.cpu())
    
    return torch.cat(all_features, dim=0)


def extract_text_features(
    model,
    texts: List[str],
    tokenizer,
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = 64,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extract text features from a model.
    
    Args:
        model: Vision-language model
        texts: List of text strings
        tokenizer: Tokenizer for the model
        device: Device to use
        batch_size: Batch size for processing
        max_length: Maximum token length
        normalize: Whether to L2-normalize features
        
    Returns:
        Tensor of text features (N, D)
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Get text features
            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if normalize:
                text_features = F.normalize(text_features, p=2, dim=-1)
            
            all_features.append(text_features.cpu())
    
    return torch.cat(all_features, dim=0)


def compute_similarity_matrix(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute similarity matrix between image and text features.
    
    Args:
        image_features: (N, D) image features
        text_features: (M, D) text features
        
    Returns:
        (N, M) similarity matrix
    """
    return torch.matmul(image_features, text_features.t())


def format_results_table(results: Dict[str, Any], title: str = "EVALUATION RESULTS") -> str:
    """
    Format results as a readable table.
    
    Args:
        results: Results dictionary
        title: Title for the table
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{title:^80}")
    lines.append("=" * 80)
    
    if "model" in results:
        lines.append(f"Model: {results['model']}")
    if "timestamp" in results:
        lines.append(f"Time: {results['timestamp']}")
    lines.append("")
    
    for dataset_name, dataset_results in results.get("results", {}).items():
        for task_name, metrics in dataset_results.items():
            lines.append(f"{task_name.upper()} - {dataset_name}")
            lines.append("-" * 80)
            
            # Check for errors first
            if "error" in metrics:
                lines.append(f"  Error: {metrics['error']}")
            elif task_name == "retrieval":
                # Format retrieval metrics
                i2t_line = f"  Image→Text:  R@1={metrics.get('i2t_r1', 0):.1f}%  R@5={metrics.get('i2t_r5', 0):.1f}%  R@10={metrics.get('i2t_r10', 0):.1f}%"
                t2i_line = f"  Text→Image:  R@1={metrics.get('t2i_r1', 0):.1f}%  R@5={metrics.get('t2i_r5', 0):.1f}%  R@10={metrics.get('t2i_r10', 0):.1f}%"
                mean_line = f"  Mean Recall: {metrics.get('mean_recall', 0):.1f}%"
                lines.extend([i2t_line, t2i_line, mean_line])
            
            elif task_name == "zero_shot":
                # Format zero-shot metrics
                for key, value in metrics.items():
                    if key == "error":
                        continue  # Already handled above
                    if isinstance(value, (int, float)):
                        lines.append(f"  {key}: {value:.1f}%")
                    else:
                        lines.append(f"  {key}: {value}")
            
            else:
                # Generic formatting
                for key, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")
            
            lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def save_results_json(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def save_results_csv(results: Dict[str, Any], filepath: str):
    """Save results to CSV file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    rows = []
    model_name = results.get("model", "unknown")
    
    for dataset_name, dataset_results in results.get("results", {}).items():
        for task_name, metrics in dataset_results.items():
            for metric_name, value in metrics.items():
                rows.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "task": task_name,
                    "metric": metric_name,
                    "value": value
                })
    
    with open(filepath, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"Results saved to {filepath}")


def save_results_latex(results: Dict[str, Any], filepath: str):
    """Save results as LaTeX table."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    lines = []
    lines.append("% Auto-generated LaTeX table")
    lines.append(f"% Generated: {results.get('timestamp', datetime.now().isoformat())}")
    lines.append("")
    
    # Retrieval table
    lines.append("% Retrieval Results")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Image-Text Retrieval Results}")
    lines.append("\\begin{tabular}{l|ccc|ccc|c}")
    lines.append("\\hline")
    lines.append("Dataset & \\multicolumn{3}{c|}{Image→Text} & \\multicolumn{3}{c|}{Text→Image} & Mean \\\\")
    lines.append("        & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 & Recall \\\\")
    lines.append("\\hline")
    
    for dataset_name, dataset_results in results.get("results", {}).items():
        if "retrieval" in dataset_results:
            m = dataset_results["retrieval"]
            line = f"{dataset_name} & {m.get('i2t_r1', 0):.1f} & {m.get('i2t_r5', 0):.1f} & {m.get('i2t_r10', 0):.1f} & "
            line += f"{m.get('t2i_r1', 0):.1f} & {m.get('t2i_r5', 0):.1f} & {m.get('t2i_r10', 0):.1f} & {m.get('mean_recall', 0):.1f} \\\\"
            lines.append(line)
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    # Zero-shot table
    lines.append("% Zero-Shot Classification Results")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Zero-Shot Classification Results}")
    lines.append("\\begin{tabular}{l|cc}")
    lines.append("\\hline")
    lines.append("Dataset & Top-1 Acc & Top-5 Acc \\\\")
    lines.append("\\hline")
    
    for dataset_name, dataset_results in results.get("results", {}).items():
        if "zero_shot" in dataset_results:
            m = dataset_results["zero_shot"]
            line = f"{dataset_name} & {m.get('top1_accuracy', 0):.1f} & {m.get('top5_accuracy', 0):.1f} \\\\"
            lines.append(line)
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(filepath, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"LaTeX tables saved to {filepath}")


def save_results(
    results: Dict[str, Any],
    output_path: str,
    formats: List[str] = ["json", "csv", "latex"],
):
    """
    Save results in multiple formats.
    
    Args:
        results: Results dictionary
        output_path: Base output path (without extension)
        formats: List of formats to save ("json", "csv", "latex")
    """
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    
    if "json" in formats:
        save_results_json(results, f"{base_path}.json")
    
    if "csv" in formats:
        save_results_csv(results, f"{base_path}.csv")
    
    if "latex" in formats:
        save_results_latex(results, f"{base_path}.tex")


def format_comparison_table(
    results_list: List[Dict[str, Any]],
    model_names: List[str],
) -> str:
    """
    Format comparison table for multiple models.
    
    Args:
        results_list: List of results dictionaries
        model_names: List of model names
        
    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'MODEL COMPARISON':^100}")
    lines.append("=" * 100)
    lines.append("")
    
    # Collect all datasets and tasks
    all_datasets = set()
    all_tasks = set()
    for results in results_list:
        for dataset_name, dataset_results in results.get("results", {}).items():
            all_datasets.add(dataset_name)
            for task_name in dataset_results.keys():
                all_tasks.add(task_name)
    
    # Retrieval comparison
    if "retrieval" in all_tasks:
        lines.append("RETRIEVAL COMPARISON")
        lines.append("-" * 100)
        
        # Header
        header = f"{'Model':<20} | {'Dataset':<15} | {'I2T R@1':>8} | {'I2T R@5':>8} | {'I2T R@10':>8} | {'T2I R@1':>8} | {'Mean':>8}"
        lines.append(header)
        lines.append("-" * 100)
        
        for model_name, results in zip(model_names, results_list):
            for dataset_name in sorted(all_datasets):
                if dataset_name in results.get("results", {}):
                    if "retrieval" in results["results"][dataset_name]:
                        m = results["results"][dataset_name]["retrieval"]
                        row = f"{model_name:<20} | {dataset_name:<15} | {m.get('i2t_r1', 0):>7.1f}% | {m.get('i2t_r5', 0):>7.1f}% | {m.get('i2t_r10', 0):>7.1f}% | {m.get('t2i_r1', 0):>7.1f}% | {m.get('mean_recall', 0):>7.1f}%"
                        lines.append(row)
        
        lines.append("")
    
    # Zero-shot comparison
    if "zero_shot" in all_tasks:
        lines.append("ZERO-SHOT CLASSIFICATION COMPARISON")
        lines.append("-" * 100)
        
        # Header
        header = f"{'Model':<20} | {'Dataset':<15} | {'Top-1 Acc':>10} | {'Top-5 Acc':>10}"
        lines.append(header)
        lines.append("-" * 100)
        
        for model_name, results in zip(model_names, results_list):
            for dataset_name in sorted(all_datasets):
                if dataset_name in results.get("results", {}):
                    if "zero_shot" in results["results"][dataset_name]:
                        m = results["results"][dataset_name]["zero_shot"]
                        row = f"{model_name:<20} | {dataset_name:<15} | {m.get('top1_accuracy', 0):>9.1f}% | {m.get('top5_accuracy', 0):>9.1f}%"
                        lines.append(row)
        
        lines.append("")
    
    lines.append("=" * 100)
    return "\n".join(lines)
