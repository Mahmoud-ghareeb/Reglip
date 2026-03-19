"""Training script for SigLIP model."""

import os
import argparse
import torch
from transformers import SiglipModel

from data import get_flickr30k_dataloaders
from data.preprocessing import create_siglip_processors
from training.trainer import SigLIPTrainer
from training.utils import load_config, setup_logging, create_optimizer_and_scheduler, set_seed


def create_model(config):
    """Create SigLIP model.

    When freeze_backbone is set, loads a RegLIPModel (from SigLIP weights)
    so that freezing logic is identical between SigLIP and RegLIP experiments.
    """
    model_name = config['model']['pretrained_model']

    if config['model'].get('freeze_backbone', False):
        from reglip.utils import load_from_transformers_siglip, freeze_backbone
        model, _ = load_from_transformers_siglip(model_name)
        model = freeze_backbone(model)
    else:
        model = SiglipModel.from_pretrained(model_name)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SigLIP model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/siglip_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/flickr30k",
        help="Path to Flickr30K dataset"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data root if provided
    if args.data_root:
        config['data']['data_root'] = args.data_root
    
    # Debug mode
    if args.debug:
        config['data']['max_samples'] = 100
        config['training']['max_epochs'] = 2
        config['logging']['use_wandb'] = False
        print("Running in debug mode with limited data")
    
    # Setup logging — store logs inside the checkpoint directory
    checkpoint_dir = config['checkpointing']['checkpoint_dir']
    log_dir = os.path.join(checkpoint_dir, 'logs')
    logger = setup_logging(
        log_dir=log_dir,
        experiment_name=config['logging']['experiment_name']
    )
    
    logger.info("Starting SigLIP training")
    logger.info(f"Configuration: {config}")
    
    # Device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Validate dataset before proceeding
    logger.info("Validating dataset...")
    from data.preprocessing import validate_flickr30k_dataset, quick_data_check
    
    # Quick check first
    if not quick_data_check(config['data']['data_root']):
        logger.error("Quick dataset check failed!")
        logger.error("Please ensure your dataset is properly set up:")
        logger.error(f"  - Data root: {config['data']['data_root']}")
        logger.error("  - Expected structure:")
        logger.error("    flickr30k/")
        logger.error("    ├── images/")
        logger.error("    │   ├── image1.jpg")
        logger.error("    │   └── ...")
        logger.error("    └── captions.txt")
        raise ValueError("Dataset validation failed")
    
    # Comprehensive validation
    validation_results = validate_flickr30k_dataset(config['data']['data_root'], verbose=True)
    
    if not validation_results['valid']:
        logger.error("Dataset validation failed!")
        for error in validation_results['errors']:
            logger.error(f"  - {error}")
        raise ValueError("Dataset validation failed")
    
    # Log dataset statistics
    stats = validation_results['statistics']
    summary = validation_results['summary']
    logger.info(f"Dataset validated successfully!")
    logger.info(f"Total usable samples: {summary['total_usable_samples']}")
    logger.info(f"Estimated train/val/test: {summary['estimated_splits']['train']}/{summary['estimated_splits']['val']}/{summary['estimated_splits']['test']}")
    
    # Create data processors
    image_processor, text_processor = create_siglip_processors(
        config['model']['pretrained_model']
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_flickr30k_dataloaders(
        data_root=config['data']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_processor=image_processor,
        text_processor=text_processor,
        use_regression_targets=False,  # SigLIP uses binary targets
        max_samples=config['data'].get('max_samples'),
    )
    
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
    
    # Create model
    model = create_model(config)
    logger.info(f"Created SigLIP model")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    logger.info(f"Created optimizer: {config['training']['optimizer']}")
    logger.info(f"Created scheduler: {config['training']['scheduler']}")
    
    # Create trainer
    trainer = SigLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        logger=logger,
        device=device,
    )
    
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 