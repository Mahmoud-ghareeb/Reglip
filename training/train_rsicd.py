"""Training script for RegLIP/SigLIP on RSICD (Remote Sensing domain)."""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reglip import RegLIPModel, RegLIPConfig
from reglip.utils import load_from_transformers_siglip, setup_frozen_text_encoder
from evaluation.datasets.rsicd import RSICDTrainingDataset
from training.trainer import RegLIPTrainer, SigLIPTrainer
from training.utils import load_config, setup_logging, create_optimizer_and_scheduler, set_seed


def create_collate_fn(tokenizer, similarity_generator=None, use_regression_targets=False, max_length=64):
    """Create a collate function for the RSICD dataloader."""

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        captions = [item["caption"] for item in batch]

        # Tokenize captions
        text_inputs = tokenizer(
            captions,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # SigLIP tokenizer may not return attention_mask with max_length padding
        attention_mask = text_inputs.get(
            "attention_mask",
            torch.ones_like(text_inputs["input_ids"]),
        )

        result = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": attention_mask,
            "captions": captions,
        }

        # Pass raw image bytes for cross-modal teacher embeddings
        if "raw_image_bytes" in batch[0]:
            result["raw_images"] = [item["raw_image_bytes"] for item in batch]

        # Generate regression targets if needed
        if use_regression_targets and similarity_generator is not None:
            similarity_matrix = similarity_generator(captions)
            result["similarity_targets"] = similarity_matrix

        return result

    return collate_fn


def create_model(config, device):
    """Create model based on config."""
    model_name = config["model"]["pretrained_model"]
    loss_type = config["training"]["loss_type"]

    # Load from pretrained SigLIP
    model, reglip_config = load_from_transformers_siglip(model_name)

    # Set up frozen text encoder for RegLIP
    if loss_type == "reglip_regression":
        reglip_cfg = config["model"]["reglip_config"]
        encoder_name = reglip_cfg["frozen_text_encoder"]
        encoder_kwargs = reglip_cfg.get("encoder_kwargs", {})

        # Legacy Qwen config keys -> new kwargs
        if encoder_name in ("qwen", "qwen_api") and not encoder_kwargs:
            encoder_kwargs = {
                "base_url": reglip_cfg.get("qwen_api_url", "http://212.41.29.82:6010"),
                "model": reglip_cfg.get("qwen_model", "Qwen/Qwen3-Embedding-8B"),
            }

        setup_frozen_text_encoder(model, encoder_name=encoder_name, **encoder_kwargs)

    # Freeze backbone if configured
    if config["model"].get("freeze_backbone", False):
        from reglip.utils import freeze_backbone
        model = freeze_backbone(model)

    return model, reglip_config


def main():
    parser = argparse.ArgumentParser(description="Train on RSICD dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (reglip_rsicd_config.yaml or siglip_rsicd_config.yaml)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to RSICD dataset (overrides config)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    if args.data_root:
        config["data"]["data_root"] = args.data_root

    if args.debug:
        config["data"]["max_samples"] = 100
        config["training"]["max_epochs"] = 2

    # Setup logging
    checkpoint_dir = config["checkpointing"]["checkpoint_dir"]
    log_dir = os.path.join(checkpoint_dir, "logs")
    logger = setup_logging(
        log_dir=log_dir,
        experiment_name=config["logging"]["experiment_name"],
    )

    logger.info(f"Training {config['model']['name']} on RSICD")
    logger.info(f"Loss type: {config['training']['loss_type']}")

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Validate RSICD dataset
    data_root = config["data"]["data_root"]
    train_csv = os.path.join(data_root, "train.csv")

    if not os.path.exists(train_csv):
        logger.error(f"RSICD train.csv not found: {train_csv}")
        logger.error("Expected structure:")
        logger.error(f"  {data_root}/")
        logger.error("  ├── train.csv")
        logger.error("  ├── test.csv")
        logger.error("  └── valid.csv")
        raise FileNotFoundError(f"Missing {train_csv}")

    # Create datasets
    train_dataset = RSICDTrainingDataset(
        data_root=data_root,
        split="train",
        max_samples=config["data"].get("max_samples"),
    )
    val_dataset = RSICDTrainingDataset(
        data_root=data_root,
        split="val",
        random_caption=False,
        max_samples=config["data"].get("max_samples"),
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model, reglip_config = create_model(config, device)

    # Get tokenizer
    from transformers import SiglipProcessor
    processor = SiglipProcessor.from_pretrained(config["model"]["pretrained_model"])
    tokenizer = processor.tokenizer

    # Create collate function
    # Note: similarity targets are generated online in the model's forward pass
    # via generate_similarity_targets(), not in the collate function.
    collate_fn = create_collate_fn(
        tokenizer=tokenizer,
        max_length=config["data"].get("max_text_length", 64),
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
    )

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # Create trainer
    loss_type = config["training"]["loss_type"]
    if loss_type == "reglip_regression":
        trainer = RegLIPTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            logger=logger,
            device=device,
        )
    else:
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

    # Train
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
