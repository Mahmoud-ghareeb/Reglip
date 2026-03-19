"""Configuration classes for RegLIP model."""

from typing import Dict, Any


class RegLIPTextConfig:
    """Configuration class for RegLIP text model."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 64,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        pad_token_id: int = 1,
        bos_token_id: int = 49406,
        eos_token_id: int = 49407,
        projection_size: int = 512,
        hidden_act: str = "gelu_pytorch_tanh",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.projection_size = projection_size
        self.hidden_act = hidden_act
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class RegLIPVisionConfig:
    """Configuration class for RegLIP vision model."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        projection_size: int = 512,
        hidden_act: str = "gelu_pytorch_tanh",
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.projection_size = projection_size
        self.hidden_act = hidden_act
        
        # Computed properties
        self.num_patches = (image_size // patch_size) ** 2
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class RegLIPConfig:
    """Configuration class for RegLIP model."""
    
    def __init__(
        self,
        text_config: RegLIPTextConfig = None,
        vision_config: RegLIPVisionConfig = None,
        projection_dim: int = 768,
        logit_scale_init_value: float = 1.0,
        logit_bias_init_value: float = 0.0,
        similarity_loss_weight: float = 1.0,
        frozen_text_encoder_name: str = "qwen",
        **kwargs
    ):
        # Handle text_config - can be dict or RegLIPTextConfig object
        if text_config is None:
            self.text_config = RegLIPTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = RegLIPTextConfig(**text_config)
        elif hasattr(text_config, '__dict__'):
            # If it's a config object, extract its dict representation
            self.text_config = RegLIPTextConfig(**text_config.__dict__)
        else:
            self.text_config = text_config
            
        # Handle vision_config - can be dict or RegLIPVisionConfig object
        if vision_config is None:
            self.vision_config = RegLIPVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = RegLIPVisionConfig(**vision_config)
        elif hasattr(vision_config, '__dict__'):
            # If it's a config object, extract its dict representation
            self.vision_config = RegLIPVisionConfig(**vision_config.__dict__)
        else:
            self.vision_config = vision_config
            
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.logit_bias_init_value = logit_bias_init_value
        self.similarity_loss_weight = similarity_loss_weight
        self.frozen_text_encoder_name = frozen_text_encoder_name
        
        # Ensure projection sizes match
        if hasattr(self.text_config, 'projection_size'):
            self.text_config.projection_size = projection_dim
        if hasattr(self.vision_config, 'projection_size'):
            self.vision_config.projection_size = projection_dim
            
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_pretrained_siglip(cls, config_dict: Dict[str, Any]) -> "RegLIPConfig":
        """Create RegLIP config from SigLIP config dictionary."""
        
        # Extract text and vision config from SigLIP format
        text_config = config_dict.get("text_config", {})
        vision_config = config_dict.get("vision_config", {})
        
        # Map SigLIP config to RegLIP config
        reglip_config = {
            "text_config": text_config,
            "vision_config": vision_config,
            "projection_dim": text_config.get("projection_size", 512),
        }
        
        return cls(**reglip_config) 