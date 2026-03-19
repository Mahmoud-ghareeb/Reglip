"""Image-Text Retrieval evaluation task."""

import os
import csv

import torch
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .base import BaseTask
from ..metrics import compute_retrieval_metrics, compute_similarity_statistics
from ..utils import extract_text_features, compute_similarity_matrix


class RetrievalTask(BaseTask):
    """Image-Text Retrieval evaluation task."""
    
    name = "retrieval"
    
    def run(
        self,
        dataset,
        batch_size: int = 64,
        k_values: List[int] = [1, 5, 10],
        compute_stats: bool = True,
        examples_output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Run retrieval evaluation.
        
        Args:
            dataset: Dataset with images and captions
            batch_size: Batch size for feature extraction
            k_values: K values for Recall@K
            compute_stats: Whether to compute similarity statistics
            examples_output_path: Optional CSV path for per-example predictions
            
        Returns:
            Dictionary with retrieval metrics
        """
        model = self.evaluator.model
        tokenizer = self.evaluator.tokenizer
        device = self.evaluator.device
        
        model.eval()
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Extract image features and collect captions
        all_image_features = []
        all_captions = []
        
        print(f"Extracting features from {len(dataset)} samples...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                # Get pixel values
                pixel_values = batch['pixel_values'].to(device)
                
                # Extract image features
                image_features = model.get_image_features(pixel_values)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                all_image_features.append(image_features.cpu())
                
                # Collect captions
                if 'caption' in batch:
                    all_captions.extend(batch['caption'])
                elif 'captions' in batch:
                    all_captions.extend(batch['captions'])
        
        # Stack image features
        image_features = torch.cat(all_image_features, dim=0)
        
        # Extract text features
        print(f"Extracting text features for {len(all_captions)} captions...")
        text_features = extract_text_features(
            model=model,
            texts=all_captions,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            normalize=True,
        )
        
        # Compute similarity matrix
        print("Computing similarity matrix...")
        similarity_matrix = compute_similarity_matrix(image_features, text_features)
        
        # Compute retrieval metrics
        print("Computing retrieval metrics...")
        metrics = compute_retrieval_metrics(similarity_matrix, k_values=k_values)
        
        # Optionally compute similarity statistics
        if compute_stats:
            stats = compute_similarity_statistics(similarity_matrix)
            metrics.update({f"sim_{k}": v for k, v in stats.items()})
        
        # Optionally save per-example predictions
        if examples_output_path is not None:
            self._save_examples_csv(
                examples_output_path,
                similarity_matrix,
                dataset,
                all_captions,
                k_values=k_values,
            )
        
        return metrics
    
    def _save_examples_csv(
        self,
        output_path: str,
        similarity_matrix: torch.Tensor,
        dataset,
        captions: List[str],
        k_values: List[int],
    ) -> None:
        """
        Save per-example retrieval predictions and ground truth to CSV.
        
        Each row corresponds to one (image, caption) pair (same index).
        Records both image-to-text and text-to-image ranks and top-1 predictions.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        if isinstance(similarity_matrix, torch.Tensor):
            sim = similarity_matrix.cpu().numpy()
        else:
            sim = similarity_matrix
        
        num_samples = sim.shape[0]
        max_k = max(k_values) if k_values else 10
        
        fieldnames = [
            "index",
            "image_id",
            "image_path",
            "correct_caption",
            "i2t_rank",
            "i2t_correct@1",
            "i2t_correct@5",
            "i2t_correct@10",
            "i2t_top1_index",
            "i2t_top1_caption",
            "t2i_rank",
            "t2i_correct@1",
            "t2i_correct@5",
            "t2i_correct@10",
            "t2i_top1_index",
        ]
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(num_samples):
                # Image→Text
                sims_row = sim[i]
                sorted_row = sims_row.argsort()[::-1]  # descending
                i2t_rank = int((sorted_row == i).nonzero()[0]) if (sorted_row == i).any() else -1
                i2t_top1_index = int(sorted_row[0])
                
                # Text→Image
                sims_col = sim[:, i]
                sorted_col = sims_col.argsort()[::-1]
                t2i_rank = int((sorted_col == i).nonzero()[0]) if (sorted_col == i).any() else -1
                t2i_top1_index = int(sorted_col[0])
                
                sample = dataset.samples[i] if hasattr(dataset, "samples") else {}
                image_id = sample.get("image_id", str(i))
                image_path = sample.get("image_path", "")
                correct_caption = captions[i] if i < len(captions) else ""
                
                def in_top(rank: int, k: int) -> int:
                    return int(rank >= 0 and rank < k)
                
                row = {
                    "index": i,
                    "image_id": image_id,
                    "image_path": image_path,
                    "correct_caption": correct_caption,
                    "i2t_rank": i2t_rank + 1 if i2t_rank >= 0 else -1,
                    "i2t_correct@1": in_top(i2t_rank, 1),
                    "i2t_correct@5": in_top(i2t_rank, 5),
                    "i2t_correct@10": in_top(i2t_rank, 10),
                    "i2t_top1_index": i2t_top1_index,
                    "i2t_top1_caption": captions[i2t_top1_index] if i2t_top1_index < len(captions) else "",
                    "t2i_rank": t2i_rank + 1 if t2i_rank >= 0 else -1,
                    "t2i_correct@1": in_top(t2i_rank, 1),
                    "t2i_correct@5": in_top(t2i_rank, 5),
                    "t2i_correct@10": in_top(t2i_rank, 10),
                    "t2i_top1_index": t2i_top1_index,
                }
                
                writer.writerow(row)
