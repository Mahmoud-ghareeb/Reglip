"""Main Evaluator class for unified evaluation."""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import torch

from .tasks import RetrievalTask, ZeroShotTask
from .datasets import (
    Flickr30KRetrievalDataset, 
    CIFAR10Dataset, 
    CIFAR100Dataset,
    ImageNetDataset,
    ImageNetV2Dataset,
    ImageNetReaLDataset,
    ObjectNetDataset,
    COCORetrievalDataset,
)
from .utils import (
    format_results_table,
    format_comparison_table,
    save_results,
)


class Evaluator:
    """
    Unified evaluator for vision-language models.
    
    Supports:
    - Image-Text Retrieval (Flickr30K)
    - Zero-Shot Classification (CIFAR-10, CIFAR-100)
    - Multiple output formats (JSON, CSV, LaTeX)
    - Model comparison
    """
    
    # Available tasks
    TASKS = {
        "retrieval": RetrievalTask,
        "zero_shot": ZeroShotTask,
    }
    
    # Available datasets
    DATASETS = {
        # Retrieval datasets
        "flickr30k": {
            "class": Flickr30KRetrievalDataset,
            "task_type": "retrieval",
        },
        "coco": {
            "class": COCORetrievalDataset,
            "task_type": "retrieval",
        },
        # Classification datasets
        "cifar10": {
            "class": CIFAR10Dataset,
            "task_type": "classification",
        },
        "cifar100": {
            "class": CIFAR100Dataset,
            "task_type": "classification",
        },
        "imagenet": {
            "class": ImageNetDataset,
            "task_type": "classification",
        },
        "imagenet_v2": {
            "class": ImageNetV2Dataset,
            "task_type": "classification",
        },
        "imagenet_real": {
            "class": ImageNetReaLDataset,
            "task_type": "classification",
        },
        "objectnet": {
            "class": ObjectNetDataset,
            "task_type": "classification",
        },
    }
    
    def __init__(
        self,
        model,
        tokenizer,
        image_processor = None,
        device: str = "cuda",
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Vision-language model
            tokenizer: Text tokenizer
            image_processor: Image processor (optional)
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Initialize tasks
        self.tasks = {
            name: task_class(self) for name, task_class in self.TASKS.items()
        }
    
    def evaluate(
        self,
        task: str,
        dataset_name: str,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on a single task and dataset.
        
        Args:
            task: Task name ('retrieval' or 'zero_shot')
            dataset_name: Dataset name ('flickr30k', 'cifar10', 'cifar100')
            dataset_kwargs: Additional arguments for dataset
            task_kwargs: Additional arguments for task
            
        Returns:
            Dictionary of metrics
        """
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.TASKS.keys())}")
        
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}")
        
        # Check task compatibility
        dataset_info = self.DATASETS[dataset_name]
        expected_task_type = "retrieval" if task == "retrieval" else "classification"
        if dataset_info["task_type"] != expected_task_type:
            raise ValueError(
                f"Dataset {dataset_name} is for {dataset_info['task_type']}, "
                f"but task {task} expects {expected_task_type}"
            )
        
        # Create dataset
        dataset_kwargs = dataset_kwargs or {}
        if "image_processor" not in dataset_kwargs:
            dataset_kwargs["image_processor"] = self.image_processor
        
        dataset = dataset_info["class"](**dataset_kwargs)
        
        # Run task
        task_kwargs = task_kwargs or {}
        task_instance = self.tasks[task]
        metrics = task_instance.run(dataset, **task_kwargs)
        
        return metrics
    
    def evaluate_all(
        self,
        datasets: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        dataset_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        task_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        examples_base_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate on multiple datasets and tasks.
        
        Args:
            datasets: List of dataset names (None for all compatible)
            tasks: List of task names (None for all)
            dataset_kwargs: Dict of dataset name -> kwargs
            task_kwargs: Dict of task name -> kwargs
            
        Returns:
            Nested dictionary: dataset -> task -> metrics
        """
        if datasets is None:
            datasets = list(self.DATASETS.keys())
        if tasks is None:
            tasks = list(self.TASKS.keys())
        
        dataset_kwargs = dataset_kwargs or {}
        task_kwargs = task_kwargs or {}
        
        results = {}
        
        for dataset_name in datasets:
            dataset_info = self.DATASETS[dataset_name]
            
            # Determine which task to use based on dataset type
            if dataset_info["task_type"] == "retrieval":
                applicable_tasks = ["retrieval"] if "retrieval" in tasks else []
            else:
                applicable_tasks = ["zero_shot"] if "zero_shot" in tasks else []
            
            if not applicable_tasks:
                continue
            
            # Check if dataset has required kwargs (especially data_root)
            ds_kwargs = dataset_kwargs.get(dataset_name, {})
            
            # Datasets that require data_root
            datasets_requiring_data_root = [
                "flickr30k", "coco", "imagenet", "imagenet_v2", 
                "imagenet_real", "objectnet"
            ]
            
            if dataset_name in datasets_requiring_data_root and not ds_kwargs.get("data_root"):
                print(f"\n{'='*60}")
                print(f"Skipping {dataset_name} - data_root not provided")
                print(f"{'='*60}")
                # Map dataset names to their environment variable names
                env_var_map = {
                    "imagenet": "IMAGENET_ROOT",
                    "imagenet_v2": "IMAGENET_V2_ROOT",
                    "imagenet_real": "IMAGENET_ROOT",  # Uses same as imagenet
                    "objectnet": "OBJECTNET_ROOT",
                    "coco": "COCO_ROOT",
                    "flickr30k": "DATA_ROOT",  # Usually uses --data_root arg
                }
                # Map dataset names to their argument names
                arg_map = {
                    "imagenet": "imagenet_root",
                    "imagenet_v2": "imagenet_v2_root", 
                    "imagenet_real": "imagenet_root",  # Uses same as imagenet
                    "objectnet": "objectnet_root",
                    "coco": "coco_root",
                    "flickr30k": "data_root",
                }
                env_var_name = env_var_map.get(dataset_name, dataset_name.upper().replace('-', '_') + "_ROOT")
                arg_name = arg_map.get(dataset_name, f"{dataset_name}_root")
                print(f"  To evaluate {dataset_name}, set {env_var_name} environment variable")
                print(f"  or use --{arg_name} command-line argument")
                results[dataset_name] = {
                    task: {"error": f"data_root not provided. Set {env_var_name} env var or use --{arg_name}"} 
                    for task in applicable_tasks
                }
                continue
            
            results[dataset_name] = {}
            
            for task in applicable_tasks:
                print(f"\n{'='*60}")
                print(f"Evaluating {task} on {dataset_name}")
                print(f"{'='*60}")
                
                try:
                    # Merge task-specific kwargs and attach examples_output_path if requested
                    base_task_kwargs = task_kwargs.get(task, {})
                    local_task_kwargs = dict(base_task_kwargs) if base_task_kwargs is not None else {}
                    
                    if examples_base_path is not None:
                        examples_output_path = f"{examples_base_path}_{dataset_name}_{task}_examples.csv"
                        # Only set if caller didn't override
                        local_task_kwargs.setdefault("examples_output_path", examples_output_path)
                    
                    metrics = self.evaluate(
                        task=task,
                        dataset_name=dataset_name,
                        dataset_kwargs=ds_kwargs,
                        task_kwargs=local_task_kwargs,
                    )
                    results[dataset_name][task] = metrics
                except Exception as e:
                    print(f"Error evaluating {task} on {dataset_name}: {e}")
                    results[dataset_name][task] = {"error": str(e)}
        
        return results
    
    def run_evaluation(
        self,
        model_path: str,
        datasets: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        output_formats: List[str] = ["json", "csv", "latex"],
        dataset_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        task_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        save_examples: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation and save results.
        
        Args:
            model_path: Path to model checkpoint (for logging)
            datasets: List of dataset names
            tasks: List of task names
            output_path: Path to save results
            output_formats: List of output formats
            dataset_kwargs: Dict of dataset name -> kwargs
            task_kwargs: Dict of task name -> kwargs
            
        Returns:
            Complete results dictionary
        """
        # Determine base path for per-example CSVs
        examples_base_path = output_path if (save_examples and output_path) else None
        
        # Run evaluation
        results = self.evaluate_all(
            datasets=datasets,
            tasks=tasks,
            dataset_kwargs=dataset_kwargs,
            task_kwargs=task_kwargs,
            examples_base_path=examples_base_path,
        )
        
        # Create full results dict
        full_results = {
            "model": model_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        
        # Print results
        print("\n" + format_results_table(full_results))
        
        # Save results
        if output_path:
            save_results(full_results, output_path, formats=output_formats)
        
        return full_results


def compare_models(
    model_paths: List[str],
    model_names: List[str],
    model_loader_fn,
    datasets: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    output_formats: List[str] = ["json", "csv", "latex"],
    dataset_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    task_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Compare multiple models.
    
    Args:
        model_paths: List of paths to model checkpoints
        model_names: List of model names for display
        model_loader_fn: Function to load model, tokenizer, processor from path
        datasets: List of dataset names
        tasks: List of task names
        output_path: Base path for saving results
        output_formats: List of output formats
        dataset_kwargs: Dict of dataset name -> kwargs
        task_kwargs: Dict of task name -> kwargs
        device: Device to use
        
    Returns:
        List of results dictionaries
    """
    all_results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n{'#'*80}")
        print(f"# Evaluating: {model_name}")
        print(f"# Path: {model_path}")
        print(f"{'#'*80}\n")
        
        # Load model
        model, tokenizer, image_processor = model_loader_fn(model_path, device)
        
        # Create evaluator
        evaluator = Evaluator(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            model_path=model_path,
            datasets=datasets,
            tasks=tasks,
            output_path=f"{output_path}_{model_name}" if output_path else None,
            output_formats=output_formats,
            dataset_kwargs=dataset_kwargs,
            task_kwargs=task_kwargs,
            save_examples=True,
        )
        
        results["model_name"] = model_name
        all_results.append(results)
    
    # Print comparison table
    print("\n" + format_comparison_table(all_results, model_names))
    
    # Save comparison
    if output_path:
        comparison_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": model_names,
            "comparison": {
                name: res["results"] for name, res in zip(model_names, all_results)
            }
        }
        save_results(comparison_results, f"{output_path}_comparison", formats=output_formats)
    
    return all_results
