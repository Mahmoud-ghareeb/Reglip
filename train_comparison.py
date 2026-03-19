#!/usr/bin/env python3
"""
Comprehensive training script for comparing SigLIP and RegLIP on Flickr30K.

This script provides an easy way to train both models and compare their performance.

Usage examples:
    # Train SigLIP only
    python train_comparison.py --model siglip --data_root ./data/flickr30k
    
    # Train RegLIP only  
    python train_comparison.py --model reglip --data_root ./data/flickr30k
    
    # Train both models for comparison
    python train_comparison.py --model both --data_root ./data/flickr30k
    
    # Debug mode with limited data
    python train_comparison.py --model both --debug
    
    # Custom configurations
    python train_comparison.py --model reglip --config configs/reglip_config.yaml --epochs 5
"""

import argparse
import os
import json
import time
from datetime import datetime
from typing import Dict, Any

import torch

from training.train_siglip import main as train_siglip_main
from training.train_reglip import main as train_reglip_main
from evaluation.metrics import evaluate_model_on_flickr30k


def create_experiment_dir(base_dir: str = "./experiments") -> str:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"siglip_vs_reglip_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def setup_configs(
    exp_dir: str,
    data_root: str,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    debug: bool = False
) -> Dict[str, str]:
    """Set up configuration files for the experiment."""
    
    configs_dir = os.path.join(exp_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    
    # Base configurations
    base_configs = {
        'siglip': 'configs/siglip_config.yaml',
        'reglip': 'configs/reglip_config.yaml'
    }
    
    config_paths = {}
    
    for model_name, base_config in base_configs.items():
        
        # Load base config
        from training.utils import load_config
        config = load_config(base_config)
        
        # Update with experiment settings
        config['data']['data_root'] = data_root
        
        # Override config values if provided
        if epochs is not None:
            config['training']['max_epochs'] = int(epochs)
        if batch_size is not None:
            config['training']['batch_size'] = int(batch_size)
        if learning_rate is not None:
            config['training']['learning_rate'] = float(learning_rate)
        
        # Debug mode settings
        if debug:
            config['data']['max_samples'] = 200  # Limit samples for faster testing
            config['training']['max_epochs'] = 2  # Shorter training
            config['training']['batch_size'] = 8  # Smaller batch
            config['data']['num_workers'] = 0  # Disable multiprocessing to avoid CUDA issues
            config['data']['pin_memory'] = False  # Disable pin_memory to avoid CUDA tensor issues
            config['logging']['use_wandb'] = False
            config['logging']['log_every_n_steps'] = 10
        
        # Update paths for this experiment
        config['checkpointing']['checkpoint_dir'] = os.path.join(exp_dir, f"checkpoints_{model_name}")
        config['logging']['experiment_name'] = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save modified config
        config_path = os.path.join(configs_dir, f"{model_name}_config.yaml")
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        config_paths[model_name] = config_path
    
    return config_paths


def train_model(model_type: str, config_path: str, exp_dir: str) -> Dict[str, Any]:
    """Train a specific model and return results."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if model_type == 'siglip':
            # Modify sys.argv to pass config to training script
            import sys
            original_argv = sys.argv.copy()
            sys.argv = ['train_siglip.py', '--config', config_path]
            
            train_siglip_main()
            
            # Restore original argv
            sys.argv = original_argv
            
        elif model_type == 'reglip':
            # Modify sys.argv to pass config to training script
            import sys
            original_argv = sys.argv.copy()
            sys.argv = ['train_reglip.py', '--config', config_path]
            
            train_reglip_main()
            
            # Restore original argv
            sys.argv = original_argv
        
        training_time = time.time() - start_time
        
        print(f"\n{model_type.upper()} training completed successfully!")
        print(f"Training time: {training_time/3600:.2f} hours")
        
        return {
            'success': True,
            'training_time': training_time,
            'error': None
        }
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"\n{model_type.upper()} training failed: {e}")
        
        return {
            'success': False,
            'training_time': training_time,
            'error': str(e)
        }


def save_experiment_summary(
    exp_dir: str,
    results: Dict[str, Any],
    config_paths: Dict[str, str],
    args: argparse.Namespace
) -> None:
    """Save experiment summary and results."""
    
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(results.keys()),
            'data_root': args.data_root,
            'debug_mode': args.debug,
        },
        'arguments': vars(args),
        'config_paths': config_paths,
        'results': results,
    }
    
    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment summary saved to: {summary_path}")


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print a summary of training results."""
    
    print(f"\n{'='*60}")
    print("TRAINING RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Status: {'✓ Success' if result['success'] else '✗ Failed'}")
        print(f"  Training time: {result['training_time']/3600:.2f} hours")
        
        if not result['success']:
            print(f"  Error: {result['error']}")
    
    # Overall summary
    successful_models = [name for name, result in results.items() if result['success']]
    total_time = sum(result['training_time'] for result in results.values())
    
    print(f"\nOVERALL:")
    print(f"  Successful models: {len(successful_models)}/{len(results)}")
    print(f"  Total training time: {total_time/3600:.2f} hours")
    
    if len(successful_models) > 1:
        print(f"\n🎉 Both models trained successfully! You can now compare their performance.")
        print(f"   Check the experiment directory for checkpoints and logs.")


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare SigLIP and RegLIP models on Flickr30K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model",
        choices=["siglip", "reglip", "both"],
        default="both",
        help="Which model(s) to train"
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/mahmoud/RegLIP/data/flickr30k",
        help="Path to Flickr30K dataset directory"
    )
    
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Experiment directory (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited data"
    )
    
    args = parser.parse_args()
    
    # Create experiment directory
    if args.exp_dir is None:
        exp_dir = create_experiment_dir()
    else:
        exp_dir = args.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_root):
        print(f"\nWarning: Data directory {args.data_root} does not exist.")
        print("The training will create dummy data for testing purposes.")
        print("For real training, please download Flickr30K dataset.")
    else:
        # Validate dataset comprehensively
        print(f"\n🔍 Validating dataset at {args.data_root}...")
        from data.preprocessing import validate_flickr30k_dataset, quick_data_check
        
        # Quick check first
        if not quick_data_check(args.data_root):
            print("❌ Quick dataset check failed!")
            print("Please ensure your dataset is properly set up:")
            print(f"  - Data root: {args.data_root}")
            print("  - Expected structure:")
            print("    flickr30k/")
            print("    ├── images/")
            print("    │   ├── image1.jpg")
            print("    │   └── ...")
            print("    └── captions.txt")
            print("\nYou can run 'python validate_dataset.py' for detailed validation.")
            if not args.debug:
                print("Aborting training. Use --debug to continue with dummy data.")
                return 1
            else:
                print("Debug mode: Continuing with dummy data...")
        else:
            # Comprehensive validation in quiet mode for cleaner output
            validation_results = validate_flickr30k_dataset(args.data_root, verbose=False)
            
            if validation_results['valid']:
                summary = validation_results['summary']
                print(f"✅ Dataset validated - {summary['total_usable_samples']} usable samples")
                splits = summary['estimated_splits']
                print(f"📊 Estimated splits - Train: {splits['train']}, Val: {splits['val']}, Test: {splits['test']}")
            else:
                print("❌ Dataset validation failed:")
                for error in validation_results['errors']:
                    print(f"  - {error}")
                print("\nRun 'python validate_dataset.py' for detailed validation.")
                if not args.debug:
                    print("Aborting training. Use --debug to continue anyway.")
                    return 1
                else:
                    print("Debug mode: Continuing despite validation errors...")
    
    # Set up configurations
    config_paths = setup_configs(
        exp_dir=exp_dir,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        debug=args.debug
    )
    
    print(f"Configurations prepared in: {os.path.join(exp_dir, 'configs')}")
    
    # Determine which models to train
    if args.model == "both":
        models_to_train = ["siglip", "reglip"]
    else:
        models_to_train = [args.model]
    
    print(f"Models to train: {', '.join(models_to_train)}")
    
    # Train models
    results = {}
    total_start_time = time.time()
    
    for model_name in models_to_train:
        config_path = config_paths[model_name]
        result = train_model(model_name, config_path, exp_dir)
        results[model_name] = result
        
        # Stop if training failed and not in debug mode
        if not result['success'] and not args.debug:
            print(f"Stopping due to {model_name} training failure.")
            break
    
    total_time = time.time() - total_start_time
    
    # Save experiment summary
    save_experiment_summary(exp_dir, results, config_paths, args)
    
    # Print results summary
    print_results_summary(results)
    
    print(f"\nTotal experiment time: {total_time/3600:.2f} hours")
    print(f"All results saved to: {exp_dir}")
    
    # Final recommendations
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    
    successful_models = [name for name, result in results.items() if result['success']]
    
    if len(successful_models) > 0:
        print(f"✓ Check model checkpoints in: {exp_dir}/checkpoints_*/")
        print(f"✓ Review training logs in: ./logs/")
        
        if len(successful_models) > 1:
            print(f"✓ Run evaluation script to compare models:")
            print(f"  python evaluate_models.py --exp_dir {exp_dir}")
    
    if args.debug:
        print(f"✓ Run full training without --debug flag for real results")
    
    print(f"✓ Monitor training with TensorBoard or W&B (if enabled)")


if __name__ == "__main__":
    main() 