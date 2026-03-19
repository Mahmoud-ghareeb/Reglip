"""Utilities for RegLIP model, particularly for loading SigLIP checkpoints."""

import json
import math
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from transformers import SiglipModel, SiglipConfig

from .config import RegLIPConfig, RegLIPTextConfig, RegLIPVisionConfig
from .model import RegLIPModel


def load_siglip_config(model_name_or_path: str) -> Dict[str, Any]:
    """Load SigLIP configuration from transformers."""
    try:
        siglip_config = SiglipConfig.from_pretrained(model_name_or_path)
        return siglip_config.to_dict()
    except Exception as e:
        print(f"Error loading SigLIP config from {model_name_or_path}: {e}")
        # Return default config
        return {
            "text_config": {
                "vocab_size": 32000,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "max_position_embeddings": 64,
                "hidden_act": "gelu_pytorch_tanh",
                "layer_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "pad_token_id": 1,
                "bos_token_id": 49406,
                "eos_token_id": 49407,
                "projection_size": 768,
            },
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "num_channels": 3,
                "image_size": 224,
                "patch_size": 16,
                "hidden_act": "gelu_pytorch_tanh",
                "layer_norm_eps": 1e-6,
                "attention_dropout": 0.0,
                "projection_size": 768,
            },
            "projection_dim": 768,
            "logit_scale_init_value": 1.0,
            "logit_bias_init_value": 0.0,
        }


def convert_siglip_config_to_reglip(siglip_config_dict: Dict[str, Any]) -> RegLIPConfig:
    """Convert SigLIP config to RegLIP config."""
    text_config = siglip_config_dict.get("text_config", {})
    vision_config = siglip_config_dict.get("vision_config", {})
    
    # Extract projection size from text config (SigLIP stores it there)
    if hasattr(text_config, '__dict__'):
        projection_dim = getattr(text_config, 'projection_size', 768)
    else:
        projection_dim = text_config.get("projection_size", 768)
    
    reglip_config = RegLIPConfig(
        text_config=text_config,  # Pass directly - RegLIPConfig will handle the conversion
        vision_config=vision_config,  # Pass directly - RegLIPConfig will handle the conversion
        projection_dim=projection_dim,
        logit_scale_init_value=siglip_config_dict.get("logit_scale_init_value", 1.0),
        logit_bias_init_value=siglip_config_dict.get("logit_bias_init_value", 0.0),
    )
    
    return reglip_config


def map_siglip_state_dict_to_reglip(
    siglip_state_dict: Dict[str, torch.Tensor],
    reglip_config: RegLIPConfig
) -> Dict[str, torch.Tensor]:
    """Map SigLIP state dict keys to RegLIP format."""
    reglip_state_dict = {}
    
    for key, value in siglip_state_dict.items():
        # Remove 'siglip.' prefix if present
        if key.startswith("siglip."):
            key = key[7:]  # Remove 'siglip.'
            
        # Map text model keys
        if key.startswith("text_model."):
            # SigLIP: text_model.embeddings.token_embedding.weight
            # RegLIP: text_model.text_model.embeddings.token_embedding.weight
            new_key = key.replace("text_model.", "text_model.text_model.")
            reglip_state_dict[new_key] = value
            
        # Map vision model keys
        elif key.startswith("vision_model."):
            # SigLIP: vision_model.embeddings.patch_embedding.weight
            # RegLIP: vision_model.vision_model.embeddings.patch_embedding.weight
            new_key = key.replace("vision_model.", "vision_model.vision_model.")
            reglip_state_dict[new_key] = value
            
        # Map logit scale and bias
        elif key in ["logit_scale", "logit_bias"]:
            reglip_state_dict[key] = value
            
        else:
            # Keep other keys as is
            reglip_state_dict[key] = value
    
    return reglip_state_dict


def load_siglip_checkpoint(
    model_name_or_path: str,
    device: str = "cpu"
) -> Tuple[RegLIPModel, RegLIPConfig]:
    """
    Load a RegLIP model from a SigLIP checkpoint.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        device: Device to load the model on
        
    Returns:
        Tuple of (RegLIP model, RegLIP config)
    """
    # Load SigLIP config and convert to RegLIP
    siglip_config_dict = load_siglip_config(model_name_or_path)
    reglip_config = convert_siglip_config_to_reglip(siglip_config_dict)
    
    # Create RegLIP model
    reglip_model = RegLIPModel(reglip_config)
    
    # Load SigLIP weights
    try:
        siglip_model = SiglipModel.from_pretrained(model_name_or_path)
        siglip_state_dict = siglip_model.state_dict()
        
        # Map state dict keys
        reglip_state_dict = map_siglip_state_dict_to_reglip(siglip_state_dict, reglip_config)
        
        # Load weights into RegLIP model
        missing_keys, unexpected_keys = reglip_model.load_state_dict(reglip_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
            
        print(f"Successfully loaded RegLIP model from SigLIP checkpoint: {model_name_or_path}")
        
    except Exception as e:
        print(f"Error loading SigLIP checkpoint: {e}")
        print("Using randomly initialized RegLIP model")
    
    return reglip_model.to(device), reglip_config


def load_from_transformers_siglip(
    model_name: str = "google/siglip-base-patch16-224",
    device: str = "cpu"
) -> Tuple[RegLIPModel, RegLIPConfig]:
    """
    Load RegLIP model from a transformers SigLIP model.
    
    Args:
        model_name: SigLIP model name from transformers
        device: Device to load model on
        
    Returns:
        Tuple of (RegLIP model, RegLIP config)
    """
    print(f"Loading RegLIP model from transformers SigLIP: {model_name}")
    
    try:
        # Load SigLIP model and config
        siglip_model = SiglipModel.from_pretrained(model_name)
        siglip_config = SiglipConfig.from_pretrained(model_name)
        
        # Convert config
        siglip_config_dict = siglip_config.to_dict()
        reglip_config = convert_siglip_config_to_reglip(siglip_config_dict)
        
        # Create RegLIP model
        reglip_model = RegLIPModel(reglip_config)
        
        # Get SigLIP state dict
        siglip_state_dict = siglip_model.state_dict()
        
        # Map to RegLIP format
        reglip_state_dict = map_siglip_state_dict_to_reglip(siglip_state_dict, reglip_config)
        
        # Check for dimension mismatches and filter problematic keys
        reglip_state_dict_filtered = {}
        skipped_keys = []
        for key, value in reglip_state_dict.items():
            try:
                model_param = reglip_model.state_dict()[key]
                if model_param.shape != value.shape:
                    skipped_keys.append(f"{key}: model {model_param.shape} vs checkpoint {value.shape}")
                    continue
            except KeyError:
                # Key doesn't exist in model, will be reported as unexpected
                pass
            reglip_state_dict_filtered[key] = value
        
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys due to shape mismatch:")
            for k in skipped_keys[:5]:  # Show first 5
                print(f"  - {k}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")
        
        # Load weights
        missing_keys, unexpected_keys = reglip_model.load_state_dict(reglip_state_dict_filtered, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
            for k in missing_keys[:5]:
                print(f"  - {k}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            for k in unexpected_keys[:5]:
                print(f"  - {k}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
            
        print(f"Successfully loaded RegLIP from {model_name}")
        
        return reglip_model.to(device), reglip_config
        
    except Exception as e:
        print(f"Error loading from transformers: {e}")
        # Fallback to default config
        siglip_config_dict = load_siglip_config(model_name)
        reglip_config = convert_siglip_config_to_reglip(siglip_config_dict)
        reglip_model = RegLIPModel(reglip_config)
        
        return reglip_model.to(device), reglip_config


def setup_frozen_text_encoder(
    reglip_model: RegLIPModel,
    encoder_name: str = "qwen",
    base_url: str = "http://212.41.29.82:6010",
    model: str = "Qwen/Qwen3-Embedding-8B"
) -> RegLIPModel:
    """
    Set up the frozen text encoder for similarity target generation.
    
    Args:
        reglip_model: RegLIP model instance
        encoder_name: Type of encoder ("qwen" or sentence transformer model name)
        base_url: Base URL for Qwen API (only used if encoder_name is "qwen")
        model: Qwen model name (only used if encoder_name is "qwen")
        
    Returns:
        RegLIP model with frozen text encoder set up
    """
    try:
        if encoder_name == "qwen":
            # Use Qwen embedding client
            from .embedding_utils import QwenEmbeddingClient
            
            print(f"Loading Qwen embedding client: {model}")
            print(f"API URL: {base_url}")
            frozen_encoder = QwenEmbeddingClient(base_url=base_url, model=model)
            reglip_model.set_frozen_text_encoder(frozen_encoder)
            print("Qwen embedding client set up successfully")
            
        else:
            # Use sentence transformer
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading frozen text encoder: {encoder_name}")
            frozen_encoder = SentenceTransformer(encoder_name)
            reglip_model.set_frozen_text_encoder(frozen_encoder)
            print("Sentence transformer encoder set up successfully")
        
    except ImportError as e:
        if encoder_name == "qwen":
            print("QwenEmbeddingClient not available. Check embedding_utils.py")
        else:
            print("sentence_transformers not available. Please install it for similarity target generation.")
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error setting up frozen text encoder: {e}")
    
    return reglip_model


def save_reglip_model(
    model: RegLIPModel,
    config: RegLIPConfig,
    save_path: str
) -> None:
    """Save RegLIP model and config."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), save_path / "pytorch_model.bin")
    
    # Save config
    with open(save_path / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    print(f"RegLIP model saved to {save_path}")


def load_reglip_model(
    load_path: str,
    device: str = "cpu"
) -> Tuple[RegLIPModel, RegLIPConfig]:
    """Load RegLIP model and config."""
    load_path = Path(load_path)
    
    # Load config
    with open(load_path / "config.json", "r") as f:
        config_dict = json.load(f)
    
    config = RegLIPConfig(**config_dict)
    
    # Create and load model
    model = RegLIPModel(config)
    model.load_state_dict(torch.load(load_path / "pytorch_model.bin", map_location=device))
    
    return model.to(device), config 