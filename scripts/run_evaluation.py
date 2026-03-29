#!/usr/bin/env python3
"""
CLI script for running evaluations.

Usage:
    # Evaluate single model
    python scripts/run_evaluation.py \
        --checkpoint checkpoints/reglip/best_model.pth \
        --task retrieval \
        --dataset flickr30k \
        --output results/reglip_eval

    # Evaluate all tasks/datasets
    python scripts/run_evaluation.py \
        --checkpoint checkpoints/reglip/best_model.pth \
        --task all \
        --dataset all

    # Compare two models
    python scripts/run_evaluation.py \
        --checkpoint checkpoints/reglip/best_model.pth checkpoints/siglip/best_model.pth \
        --model_names RegLIP SigLIP \
        --compare \
        --output results/comparison
"""

import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import Evaluator, compare_models


def load_reglip_model(checkpoint_path: str, device: str = "cuda"):
    """Load RegLIP model from checkpoint."""
    from transformers import SiglipProcessor
    from reglip import RegLIPModel
    from reglip.utils import load_from_transformers_siglip
    
    # Load base model
    base_model_name = "google/siglip-base-patch16-224"
    model, config = load_from_transformers_siglip(base_model_name)
    
    # Load checkpoint weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Using BASE pretrained RegLIP model (no fine-tuning checkpoint)")
    
    model.to(device)
    model.eval()
    
    # Load processor
    processor = SiglipProcessor.from_pretrained(base_model_name)
    
    # Create image processor wrapper
    class ImageProcessor:
        def __init__(self, proc):
            self.processor = proc
        
        def __call__(self, image):
            return self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
    
    return model, processor.tokenizer, ImageProcessor(processor.image_processor)


def load_siglip_model(checkpoint_path: str, device: str = "cuda"):
    """Load SigLIP model from checkpoint."""
    from transformers import SiglipModel, SiglipProcessor
    
    base_model_name = "google/siglip-base-patch16-224"
    
    # Load model
    model = SiglipModel.from_pretrained(base_model_name)
    
    # Load checkpoint weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Using BASE pretrained SigLIP model (no fine-tuning checkpoint)")
    
    model.to(device)
    model.eval()
    
    # Load processor
    processor = SiglipProcessor.from_pretrained(base_model_name)
    
    # Create image processor wrapper
    class ImageProcessor:
        def __init__(self, proc):
            self.processor = proc
        
        def __call__(self, image):
            return self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
    
    # Wrap model to have consistent API
    class SigLIPWrapper:
        def __init__(self, model):
            self.model = model
        
        def to(self, device):
            self.model.to(device)
            return self
        
        def eval(self):
            self.model.eval()
            return self
        
        def get_image_features(self, pixel_values):
            return self.model.get_image_features(pixel_values)
        
        def get_text_features(self, input_ids, attention_mask=None):
            return self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    
    return SigLIPWrapper(model), processor.tokenizer, ImageProcessor(processor.image_processor)


def auto_detect_model_type(checkpoint_path: str) -> str:
    """Auto-detect model type from checkpoint path."""
    path_lower = checkpoint_path.lower()
    if "reglip" in path_lower:
        return "reglip"
    elif "siglip" in path_lower:
        return "siglip"
    else:
        # Default to reglip
        return "reglip"


def load_model(checkpoint_path: str, device: str = "cuda", model_type: str = None):
    """Load model based on type."""
    if model_type is None:
        model_type = auto_detect_model_type(checkpoint_path)
    
    print(f"Loading model type: {model_type}")
    
    if model_type == "reglip":
        return load_reglip_model(checkpoint_path, device)
    elif model_type == "siglip":
        return load_siglip_model(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate vision-language models")
    
    # Model arguments
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to model checkpoint(s)"
    )
    parser.add_argument(
        "--model_names", "-n",
        type=str,
        nargs="+",
        help="Name(s) for model(s) (for comparison mode)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Display name for the model in CSV output (e.g., 'RegLIP Fine-tuned')"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["reglip", "siglip", "auto"],
        default="auto",
        help="Model type (auto-detected if not specified)"
    )
    
    # Task/dataset arguments
    parser.add_argument(
        "--task", "-t",
        type=str,
        nargs="+",
        default=["all"],
        help="Task(s) to evaluate: retrieval, zero_shot, or all"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        nargs="+",
        default=["all"],
        help="Dataset(s) to evaluate: flickr30k, cifar10, cifar100, or all"
    )
    
    # Dataset-specific arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/flickr30k",
        help="Root directory for Flickr30K/COCO data"
    )
    parser.add_argument(
        "--cifar_root",
        type=str,
        default="./data",
        help="Root directory for CIFAR data"
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        default=None,
        help="Root directory for ImageNet data (uses IMAGENET_ROOT env var if not set)"
    )
    parser.add_argument(
        "--imagenet_v2_root",
        type=str,
        default=None,
        help="Root directory for ImageNet-V2 data (uses IMAGENET_V2_ROOT env var if not set)"
    )
    parser.add_argument(
        "--objectnet_root",
        type=str,
        default=None,
        help="Root directory for ObjectNet data (uses OBJECTNET_ROOT env var if not set)"
    )
    parser.add_argument(
        "--coco_root",
        type=str,
        default=None,
        help="Root directory for COCO data (uses COCO_ROOT env var if not set)"
    )
    
    # Zero-shot arguments
    parser.add_argument(
        "--templates",
        type=str,
        nargs="+",
        default=None,
        help="Prompt templates for zero-shot (e.g., 'a photo of a {}')"
    )
    parser.add_argument(
        "--no_ensemble",
        action="store_true",
        help="Disable template ensembling for zero-shot"
    )
    
    # Output arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for results (without extension)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        nargs="+",
        default=["json", "csv", "latex"],
        choices=["json", "csv", "latex"],
        help="Output format(s)"
    )
    
    # Comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models"
    )
    
    # Other arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per dataset (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Process tasks and datasets
    tasks = None if "all" in args.task else args.task
    datasets = None if "all" in args.dataset else args.dataset
    
    # Get dataset roots from args or environment variables
    imagenet_root = args.imagenet_root or os.environ.get("IMAGENET_ROOT") or "./data/imagenet-1k"
    imagenet_v2_root = args.imagenet_v2_root or os.environ.get("IMAGENET_V2_ROOT")
    objectnet_root = args.objectnet_root or os.environ.get("OBJECTNET_ROOT")
    coco_root = args.coco_root or os.environ.get("COCO_ROOT", args.data_root)
    
    # Prepare dataset kwargs
    dataset_kwargs = {
        "cifar10": {"data_root": args.cifar_root, "split": "test"},
        "cifar100": {"data_root": args.cifar_root, "split": "test"},
        "patternnet": {"split": "train"},
    }

    # Add max_samples to all datasets if specified
    if args.max_samples:
        for key in dataset_kwargs:
            dataset_kwargs[key]["max_samples"] = args.max_samples
    
    # Add ImageNet datasets if root is provided
    # Mapping: data_root/ILSVRC2012_img_val_subset/{0..999}/ = class indices (notebook structure)
    # Class names from imagenet-simple-labels for SigLIP/CLIP zero-shot compatibility
    if imagenet_root:
        dataset_kwargs["imagenet"] = {"data_root": imagenet_root, "split": "val"}
    
    if imagenet_v2_root:
        dataset_kwargs["imagenet_v2"] = {"data_root": imagenet_v2_root, "split": "matched-frequency"}
    
    if objectnet_root:
        dataset_kwargs["objectnet"] = {"data_root": objectnet_root, "split": "test"}
    
    # Add COCO if root is provided
    if coco_root:
        dataset_kwargs["coco"] = {"data_root": coco_root, "split": "test"}

    
    # Prepare task kwargs
    task_kwargs = {
        "retrieval": {"batch_size": args.batch_size},
        "zero_shot": {
            "batch_size": args.batch_size,
            "templates": args.templates,
            "use_ensemble": not args.no_ensemble,
        },
    }
    
    # Run evaluation
    if args.compare or len(args.checkpoint) > 1:
        # Comparison mode
        if args.model_names is None:
            args.model_names = [os.path.basename(os.path.dirname(p)) for p in args.checkpoint]
        
        if len(args.model_names) != len(args.checkpoint):
            raise ValueError("Number of model names must match number of checkpoints")
        
        def model_loader(path, device):
            return load_model(path, device, args.model_type if args.model_type != "auto" else None)
        
        compare_models(
            model_paths=args.checkpoint,
            model_names=args.model_names,
            model_loader_fn=model_loader,
            datasets=datasets,
            tasks=tasks,
            output_path=args.output,
            output_formats=args.format,
            dataset_kwargs=dataset_kwargs,
            task_kwargs=task_kwargs,
            device=args.device,
        )
    else:
        # Single model evaluation
        checkpoint_path = args.checkpoint[0]
        model_type = args.model_type if args.model_type != "auto" else None
        
        model, tokenizer, image_processor = load_model(checkpoint_path, args.device, model_type)
        
        evaluator = Evaluator(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=args.device,
        )
        
        # Use --model_name if provided, otherwise fall back to checkpoint path
        display_name = args.model_name if args.model_name else checkpoint_path

        evaluator.run_evaluation(
            model_path=display_name,
            datasets=datasets,
            tasks=tasks,
            output_path=args.output,
            output_formats=args.format,
            dataset_kwargs=dataset_kwargs,
            task_kwargs=task_kwargs,
        )


if __name__ == "__main__":
    main()
