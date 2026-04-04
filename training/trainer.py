"""Trainer classes for SigLIP and RegLIP models."""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .utils import (
    save_checkpoint, cleanup_old_checkpoints, get_parameter_count,
    format_time, setup_wandb, calculate_gradient_norm, get_learning_rate
)


class BaseTrainer(ABC):
    """Base trainer class for vision-language models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        logger,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.device = device
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        
        # Mixed precision
        self.use_mixed_precision = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Logging
        self.wandb = setup_wandb(config)
        
        # TensorBoard logging — stored inside the checkpoint directory
        checkpoint_dir = config['checkpointing']['checkpoint_dir']
        self.use_tensorboard = config.get('logging', {}).get('use_tensorboard', True)
        if self.use_tensorboard:
            tb_dir = os.path.join(checkpoint_dir, 'tensorboard')
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.logger.info(f"TensorBoard logging enabled: {tb_dir}")
        else:
            self.tb_writer = None
        
        # Move model to device
        self.model.to(device)
        
        # Log model info
        param_counts = get_parameter_count(model)
        self.logger.info(f"Model parameter counts: {param_counts}")
        
        # Log gradient accumulation info
        if self.gradient_accumulation_steps > 1:
            batch_size = config['training'].get('batch_size', 'unknown')
            effective_batch_size = f"{batch_size} x {self.gradient_accumulation_steps} = {batch_size * self.gradient_accumulation_steps if isinstance(batch_size, int) else 'unknown'}"
            self.logger.info(f"Gradient accumulation enabled: {self.gradient_accumulation_steps} steps")
            self.logger.info(f"Effective batch size: {effective_batch_size}")
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch. Must be implemented by subclasses."""
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        accumulation_steps = self.gradient_accumulation_steps
        
        # Zero gradients at the start of epoch
        self.optimizer.zero_grad()
        
        with tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        loss_dict = self.compute_loss(batch)
                        loss = loss_dict['total_loss']
                        # Scale loss for gradient accumulation
                        loss = loss / accumulation_steps
                else:
                    loss_dict = self.compute_loss(batch)
                    loss = loss_dict['total_loss']
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Backward pass (accumulate gradients)
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update metrics (use unscaled loss for logging)
                total_loss += loss.item() * accumulation_steps
                
                # Only update weights every accumulation_steps batches (or at end of epoch)
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    if self.use_mixed_precision:
                        # Gradient clipping
                        if self.config['training'].get('gradient_clip_norm', 0) > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['gradient_clip_norm']
                            )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Gradient clipping
                        if self.config['training'].get('gradient_clip_norm', 0) > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['gradient_clip_norm']
                            )
                        
                        self.optimizer.step()
                    
                    # Zero gradients after optimizer step
                    self.optimizer.zero_grad()
                    
                    # Step scheduler after optimizer step
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Increment step counter (counts optimizer steps, not batches)
                    self.step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}",
                    'lr': f"{get_learning_rate(self.optimizer):.6f}",
                    'accum': f"{(batch_idx % accumulation_steps) + 1}/{accumulation_steps}",
                })
                
                # Log to wandb and tensorboard (only on optimizer steps)
                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.step % self.config['logging']['log_every_n_steps'] == 0:
                        # Prepare log dict
                        log_dict = {
                            'train/loss': loss.item() * accumulation_steps,
                            'train/learning_rate': get_learning_rate(self.optimizer),
                            'train/gradient_norm': calculate_gradient_norm(self.model),
                            'train/epoch': self.epoch,
                            'train/step': self.step,
                        }
                        # Add loss components
                        for key, value in loss_dict.items():
                            if key != 'total_loss':
                                log_dict[f'train/{key}'] = value.item() if isinstance(value, torch.Tensor) else value
                        
                        # Log to wandb
                        if self.wandb:
                            self.wandb.log(log_dict, step=self.step)
                        
                        # Log to TensorBoard
                        if self.tb_writer:
                            for key, value in log_dict.items():
                                if isinstance(value, (int, float)):
                                    self.tb_writer.add_scalar(key, value, self.step)
        
        return {'train_loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        loss_dict = self.compute_loss(batch)
                        loss = loss_dict['total_loss']
                else:
                    loss_dict = self.compute_loss(batch)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
        
        val_metrics = {'val_loss': total_loss / num_batches}
        
        # Log to wandb
        if self.wandb:
            log_dict = {f'val/{k}': v for k, v in val_metrics.items()}
            self.wandb.log(log_dict, step=self.step)
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f'val/{key}', value, self.step)
        
        return val_metrics
    
    def save_model(self, is_best: bool = False) -> str:
        """Save model checkpoint with 1-based epoch numbering."""
        checkpoint_dir = self.config['checkpointing']['checkpoint_dir']
        
        display_epoch = self.epoch + 1
        checkpoint_name = f"epoch_{display_epoch}.pth"
        checkpoint_path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=display_epoch,
            step=self.step,
            loss=self.best_val_loss,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            save_best=is_best,
        )
        
        # Cleanup old checkpoints
        keep_last_n = self.config['checkpointing'].get('keep_last_n_checkpoints')
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
        
        return checkpoint_path
    
    def train(self) -> None:
        """Main training loop."""
        max_epochs = self.config['training']['max_epochs']
        eval_every_n_epochs = self.config['evaluation']['eval_every_n_epochs']
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            display_epoch = epoch + 1
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = None
            is_best = False
            if epoch % eval_every_n_epochs == 0:
                val_metrics = self.validate()
                val_loss = val_metrics['val_loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Save checkpoint for every epoch + mark best
            checkpoint_path = self.save_model(is_best=is_best)
            self.logger.info(f"Saved epoch {display_epoch} checkpoint: {checkpoint_path}")
            if is_best:
                self.logger.info(f"Saved best_model.pth (epoch {display_epoch})")
            
            # Log metrics
            msg = f"Epoch {display_epoch}/{max_epochs}: Train Loss: {train_metrics['train_loss']:.4f}"
            if val_metrics:
                msg += f", Val Loss: {val_metrics['val_loss']:.4f}"
            self.logger.info(msg)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        
        if self.wandb:
            self.wandb.finish()
        
        if self.tb_writer:
            self.tb_writer.close()


class SigLIPTrainer(BaseTrainer):
    """Trainer for SigLIP model with binary contrastive loss."""
    
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute SigLIP binary contrastive loss."""

        # Forward pass — if model is RegLIPModel, force binary contrastive loss
        kwargs = dict(
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            return_loss=True,
        )
        if hasattr(self.model, 'binary_contrastive_loss'):
            kwargs['use_regression_loss'] = False
        outputs = self.model(**kwargs)
        
        loss = outputs.loss
        
        return {
            'total_loss': loss,
            'siglip_loss': loss,
        }


class RegLIPTrainer(BaseTrainer):
    """Trainer for RegLIP model with regression-based contrastive loss."""
    
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute RegLIP regression-based contrastive loss."""
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            pixel_values=batch['pixel_values'],
            attention_mask=batch['attention_mask'],
            similarity_targets=batch.get('similarity_targets'),
            texts=batch.get('captions'),
            raw_images=batch.get('raw_images'),
            return_loss=True,
        )
        
        # RegLIP model now returns RegLIPOutput object
        loss = outputs.loss
        similarity_targets = outputs.similarity_targets
        
        # Additional metrics
        loss_dict = {
            'total_loss': loss,
            'regression_loss': loss,
        }
        
        # Log similarity statistics if available
        if similarity_targets is not None:
            loss_dict.update({
                'similarity_mean': similarity_targets.mean(),
                'similarity_std': similarity_targets.std(),
                'similarity_min': similarity_targets.min(),
                'similarity_max': similarity_targets.max(),
            })
        
        return loss_dict 