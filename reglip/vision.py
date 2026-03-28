"""Vision model components for RegLIP based on SigLIP."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RegLIPVisionConfig
from .text import get_activation_fn


class RegLIPVisionEmbeddings(nn.Module):
    """Vision embeddings for RegLIP based on SigLIP."""
    
    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,  # SigLIP uses bias=True
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        position_embeds = self.position_embedding(self.position_ids)
        embeddings = patch_embeds + position_embeds
        
        return embeddings


class RegLIPVisionAttention(nn.Module):
    """Multi-head attention for vision model."""
    
    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to q, k, v
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class RegLIPVisionMLP(nn.Module):
    """MLP for vision transformer."""
    
    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.config = config
        self.activation_fn = get_activation_fn(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class RegLIPVisionEncoderLayer(nn.Module):
    """Single transformer layer for vision model."""
    
    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = RegLIPVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = RegLIPVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        
        return outputs


class RegLIPVisionEncoder(nn.Module):
    """Vision transformer encoder."""
    
    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            RegLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[tuple], Optional[tuple]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return hidden_states, all_hidden_states, all_attentions


class RegLIPMultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-head attention pooling head matching SigLIP's architecture.
    This uses a learnable probe vector that attends to all patch tokens.
    """

    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = RegLIPVisionMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        probe = self.probe.expand(batch_size, -1, -1)

        # Attention: probe queries the hidden states
        hidden_states, _ = self.attention(probe, hidden_states, hidden_states)

        # Residual connection not used here (matches SigLIP)
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states[:, 0, :]  # Return [batch_size, hidden_size]


class RegLIPVisionTransformer(nn.Module):
    """Complete vision transformer matching SigLIP structure."""
    
    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = RegLIPVisionEmbeddings(config)
        self.encoder = RegLIPVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Use multi-head attention pooling head like SigLIP
        self.head = RegLIPMultiheadAttentionPoolingHead(config)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        interpolate_pos_encoding: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get patch embeddings
        hidden_states = self.embeddings(pixel_values)
        
        # Pass through encoder
        last_hidden_state, all_hidden_states, all_attentions = self.encoder(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        # Use multi-head attention pooling (like SigLIP)
        # The head takes the full sequence and uses a learnable probe to attend to it
        pooled_output = self.head(last_hidden_state)
        
        outputs = (last_hidden_state, pooled_output)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        
        return outputs


class RegLIPVisionModel(nn.Module):
    """Vision model for RegLIP based on SigLIP architecture."""
    
    def __init__(self, config: RegLIPVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = RegLIPVisionTransformer(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        interpolate_pos_encoding: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ) 