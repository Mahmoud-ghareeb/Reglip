"""Utilities for training scripts."""

import os
import random
import logging
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.optim as optim
import yaml
import wandb


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer_and_scheduler(model, config):
    """Create optimizer and learning rate scheduler."""
    
    # Optimizer
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Scheduler
    scheduler_name = config['training']['scheduler'].lower()
    warmup_steps = config['training']['warmup_steps']
    max_epochs = config['training']['max_epochs']
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=0
        )
    elif scheduler_name == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
    elif scheduler_name == 'constant':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return optimizer, scheduler


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(
    log_dir: str = "./logs",
    experiment_name: str = "experiment",
    level: int = logging.INFO
) -> logging.Logger:
    """Set up logging for training."""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    checkpoint_name: str = "checkpoint.pth",
    save_best: bool = False,
) -> str:
    """Save model checkpoint."""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if save_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
    }


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: Optional[int] = 3):
    """Remove old checkpoints, keeping only the last N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep. If None or <= 0, keeps all checkpoints.
    """
    
    if not os.path.exists(checkpoint_dir):
        return
    
    # Skip cleanup if keep_last_n is None or <= 0 (keep all checkpoints)
    if keep_last_n is None or keep_last_n <= 0:
        return
    
    # Find all checkpoint files (supports both epoch_N.pth and checkpoint_epoch_N.pth)
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file == "best_model.pth":
            continue
        if file.startswith("epoch_") and file.endswith(".pth"):
            epoch_num = int(file.split("_")[1].split(".")[0])
            checkpoint_files.append((epoch_num, file))
        elif file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
            epoch_num = int(file.split("_")[2].split(".")[0])
            checkpoint_files.append((epoch_num, file))
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[0])
    
    # Remove old checkpoints
    if len(checkpoint_files) > keep_last_n:
        files_to_remove = checkpoint_files[:-keep_last_n]
        for _, filename in files_to_remove:
            file_path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed old checkpoint: {filename}")


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """Get parameter counts for a model."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"


def setup_wandb(config: Dict[str, Any]) -> Optional[Any]:
    """Set up Weights & Biases logging with robust error handling."""
    
    if not config.get('logging', {}).get('use_wandb', False):
        return None
    
    try:
        import wandb
        
        # Try to initialize wandb with error handling
        wandb.init(
            project=config['logging'].get('project_name', 'reglip-experiment'),
            name=config['logging'].get('experiment_name', 'reglip-run'),
            config=config,
            mode="online",  # Try online first
        )
        
        print("✅ Wandb initialized successfully")
        return wandb
        
    except ImportError:
        print("⚠️  wandb not available, skipping W&B logging")
        return None
    except Exception as e:
        # Handle authentication, network, or permission errors
        if "403" in str(e) or "Forbidden" in str(e):
            print("⚠️  Wandb authentication error (403 Forbidden)")
            print("   This usually means:")
            print("   1. You're not logged in: run 'wandb login'")
            print("   2. No permission for this project")
            print("   3. Network/proxy issues")
        elif "401" in str(e) or "Unauthorized" in str(e):
            print("⚠️  Wandb authentication error (401 Unauthorized)")
            print("   Please run 'wandb login' to authenticate")
        else:
            print(f"⚠️  Wandb initialization failed: {e}")
        
        # Try offline mode as fallback
        try:
            import wandb
            print("   Trying offline mode...")
            wandb.init(
                project=config['logging'].get('project_name', 'reglip-experiment'),
                name=config['logging'].get('experiment_name', 'reglip-run'),
                config=config,
                mode="offline",
            )
            print("✅ Wandb initialized in offline mode")
            return wandb
        except Exception as offline_error:
            print(f"⚠️  Offline mode also failed: {offline_error}")
            print("   Continuing training without wandb logging")
            return None


def calculate_gradient_norm(model: torch.nn.Module) -> float:
    """Calculate the gradient norm for a model."""
    
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    return (total_norm ** 0.5) if param_count > 0 else 0.0


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0 