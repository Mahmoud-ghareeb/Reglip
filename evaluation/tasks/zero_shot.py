"""Zero-Shot Classification evaluation task."""

import os
import csv

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .base import BaseTask
from ..metrics import compute_accuracy
from ..utils import extract_text_features


class ZeroShotTask(BaseTask):
    """Zero-Shot Classification evaluation task."""
    
    name = "zero_shot"
    
    # Default prompt templates
    DEFAULT_TEMPLATES = [
        "a photo of a {}",
        "a picture of a {}",
        "an image of a {}",
        "{}",
    ]
    
    def run(
        self,
        dataset,
        batch_size: int = 64,
        k_values: List[int] = [1, 5],
        templates: Optional[List[str]] = None,
        use_ensemble: bool = True,
        examples_output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Run zero-shot classification evaluation.
        
        Args:
            dataset: Classification dataset with images and labels
            batch_size: Batch size for feature extraction
            k_values: K values for top-K accuracy
            templates: Prompt templates (uses defaults if None)
            use_ensemble: Whether to ensemble multiple templates
            examples_output_path: Optional CSV path for per-example predictions
            
        Returns:
            Dictionary with classification metrics
        """
        model = self.evaluator.model
        tokenizer = self.evaluator.tokenizer
        device = self.evaluator.device
        
        model.eval()
        
        # Get class names
        class_names = dataset.get_class_names()
        num_classes = len(class_names)
        print(f"Evaluating zero-shot classification on {num_classes} classes")
        
        # Get templates
        if templates is None:
            templates = self.DEFAULT_TEMPLATES
        
        # Create text features for all classes
        print("Creating class text features...")
        if use_ensemble:
            # Ensemble multiple templates
            class_features = self._create_ensemble_class_features(
                model, tokenizer, class_names, templates, device, batch_size
            )
        else:
            # Use single template
            class_features = self._create_single_class_features(
                model, tokenizer, class_names, templates[0], device, batch_size
            )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Evaluate
        all_predictions = []
        all_labels = []
        example_rows: List[Dict[str, Any]] = []
        global_index = 0
        
        print("Evaluating images...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Zero-shot evaluation"):
                # Get images and labels
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    images = batch.get('pixel_values', batch.get('image'))
                    labels = batch.get('label', batch.get('labels'))
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")
                
                images = images.to(device)
                
                # Extract image features
                image_features = model.get_image_features(images)
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                # Compute similarities
                similarities = torch.matmul(image_features, class_features.t())
                
                all_predictions.append(similarities.cpu())
                all_labels.append(labels)
                
                # Collect per-example predictions for CSV
                if examples_output_path is not None:
                    top1_scores, top1_indices = similarities.max(dim=1)
                    max_k = max(k_values) if k_values else 5
                    topk_scores, topk_indices = similarities.topk(min(max_k, num_classes), dim=1)
                    
                    for i in range(similarities.size(0)):
                        true_label = int(labels[i].item())
                        pred_label = int(top1_indices[i].item())
                        row = {
                            "index": global_index,
                            "true_label": true_label,
                            "true_class_name": class_names[true_label],
                            "pred_label": pred_label,
                            "pred_class_name": class_names[pred_label],
                            "correct_top1": int(true_label == pred_label),
                        }
                        # Add top-k info as comma-separated lists
                        topk_lbls = [int(idx.item()) for idx in topk_indices[i]]
                        topk_names = [class_names[j] for j in topk_lbls]
                        row["topk_labels"] = ";".join(map(str, topk_lbls))
                        row["topk_class_names"] = ";".join(topk_names)
                        example_rows.append(row)
                        global_index += 1
        
        # Stack all predictions and labels
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0) if isinstance(all_labels[0], torch.Tensor) else torch.tensor(all_labels)
        
        # Compute accuracy
        metrics = compute_accuracy(predictions, labels, k_values=k_values)
        
        # Optionally save per-example CSV
        if examples_output_path is not None and example_rows:
            os.makedirs(os.path.dirname(examples_output_path) or ".", exist_ok=True)
            fieldnames = [
                "index",
                "true_label",
                "true_class_name",
                "pred_label",
                "pred_class_name",
                "correct_top1",
                "topk_labels",
                "topk_class_names",
            ]
            with open(examples_output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(example_rows)
        
        return metrics
    
    def _create_single_class_features(
        self,
        model,
        tokenizer,
        class_names: List[str],
        template: str,
        device: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Create class features using a single template."""
        texts = [template.format(name) for name in class_names]
        
        features = extract_text_features(
            model=model,
            texts=texts,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            normalize=True,
        )
        
        return features.to(device)
    
    def _create_ensemble_class_features(
        self,
        model,
        tokenizer,
        class_names: List[str],
        templates: List[str],
        device: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Create class features by ensembling multiple templates."""
        all_features = []
        
        for template in templates:
            features = self._create_single_class_features(
                model, tokenizer, class_names, template, device, batch_size
            )
            all_features.append(features)
        
        # Average features across templates
        ensemble_features = torch.stack(all_features, dim=0).mean(dim=0)
        
        # Re-normalize
        ensemble_features = F.normalize(ensemble_features, p=2, dim=-1)
        
        return ensemble_features
