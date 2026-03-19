"""Text model components for RegLIP based on SigLIP."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RegLIPTextConfig


def get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "gelu":
        return F.gelu
    elif activation == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    elif activation == "relu":
        return F.relu
    elif activation == "silu" or activation == "swish":
        return F.silu
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class RegLIPTextEmbeddings(nn.Module):
    """Text embeddings for RegLIP."""
    
    def __init__(self, config: RegLIPTextConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Register buffer for position_ids
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        embeddings = token_embeds + position_embeds
        return embeddings


class RegLIPTextAttention(nn.Module):
    """Multi-head attention for text model."""
    
    def __init__(self, config: RegLIPTextConfig):
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
        attention_mask: Optional[torch.Tensor] = None,
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
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
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


class RegLIPTextMLP(nn.Module):
    """MLP for text transformer."""
    
    def __init__(self, config: RegLIPTextConfig):
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


class RegLIPTextEncoderLayer(nn.Module):
    """Single transformer layer for text model."""
    
    def __init__(self, config: RegLIPTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = RegLIPTextAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = RegLIPTextMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
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


class RegLIPTextEncoder(nn.Module):
    """Text transformer encoder."""
    
    def __init__(self, config: RegLIPTextConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            RegLIPTextEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[tuple, torch.Tensor]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


def _prepare_4d_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Prepare 4D attention mask for transformer."""
    batch_size, seq_len = attention_mask.shape
    # Convert to 4D mask: [batch_size, 1, tgt_seq_len, src_seq_len]
    attention_mask = attention_mask[:, None, None, :]
    attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
    
    # Convert to float and create additive mask
    attention_mask = attention_mask.to(dtype=dtype)
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
    
    return attention_mask


class RegLIPTextTransformer(nn.Module):
    """Complete text transformer matching SigLIP structure."""
    
    def __init__(self, config: RegLIPTextConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = RegLIPTextEmbeddings(config)
        self.encoder = RegLIPTextEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.projection_size)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get embeddings
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        
        # Prepare attention mask (SigLIP style)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        
        # Pool the sequence - use the last token (EOS token) for pooling like SigLIP
        if attention_mask is not None:
            # Get the original attention mask (not the 4D version)
            if input_ids is not None:
                batch_size = input_ids.shape[0]
                # Find the last non-padding token for each sequence
                # Reconstruct original mask from input_ids
                original_mask = (input_ids != self.config.pad_token_id).float()
                sequence_lengths = original_mask.sum(dim=-1) - 1  # -1 for 0-indexing
                sequence_lengths = sequence_lengths.long().clamp(min=0)
                pooled_output = last_hidden_state[range(batch_size), sequence_lengths]
            else:
                # Fallback: use last token
                pooled_output = last_hidden_state[:, -1]
        else:
            # If no attention mask, use the last token
            pooled_output = last_hidden_state[:, -1]
        
        # Apply projection head
        pooled_output = self.head(pooled_output)
        
        outputs = (last_hidden_state, pooled_output)
        if output_hidden_states and len(encoder_outputs) > 1:
            outputs = outputs + (encoder_outputs[1],)
        if output_attentions and len(encoder_outputs) > 2:
            outputs = outputs + (encoder_outputs[-1],)
        
        return outputs


class RegLIPTextModel(nn.Module):
    """Text model for RegLIP based on SigLIP architecture."""
    
    def __init__(self, config: RegLIPTextConfig):
        super().__init__()
        self.config = config
        self.text_model = RegLIPTextTransformer(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
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
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        ) 