"""
Standard Diffusion Transformer (DiT) implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Modulate the input tensor with shift and scale (for adaLN)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    Standard DiT block with adaLN-Zero.
    Supports optional gated attention.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_gated_attention: bool = False,
        gate_mode: str = "element-wise",
        mlp_type: str = "standard",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Choose attention mechanism
        if use_gated_attention:
            from additional_modules.gated_attention_SDPA import GatedMultiheadAttention
            self.attn = GatedMultiheadAttention(
                num_heads=num_heads,
                d_model=hidden_size,
                dropout=dropout,
                batch_first=True,
                gate_mode=gate_mode,
            )
        else:
            self.attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_type = mlp_type.lower()
        
        # Choose MLP type
        if mlp_type == "swiglu":
            # SwiGLU: hidden_dim = (2/3) * dim_feedforward
            # This maintains similar parameter count to standard MLP
            from additional_modules.gated_attention_SDPA import SwiGLU
            mlp_hidden_dim = int(2 * dim_feedforward / 3)
            self.mlp = SwiGLU(hidden_size, mlp_hidden_dim, bias=True)
        elif mlp_type == "standard":
            # Standard MLP: Linear -> Activation -> Dropout -> Linear
            act_layer = nn.GELU() if activation == "gelu" else nn.ReLU()
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, dim_feedforward),
                act_layer,
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, hidden_size),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unsupported mlp_type: {mlp_type}. Must be 'standard' or 'swiglu'")

        # adaLN modulation
        # Output: 6 * hidden_size -> (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Zero-Init: crucial for DiT
        # Initialize the final linear layer to zeros so the block is identity at start
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # adaLN parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-Attention
        # Note: batch_first=True used in __init__, so input should be [B, L, D]
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_result = self.attn(x_norm, x_norm, x_norm)
        # Handle both nn.MultiheadAttention (returns tuple) and GatedMultiheadAttention (returns tensor)
        attn_out = attn_result[0] if isinstance(attn_result, tuple) else attn_result
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 100,
        obs_dim: int = 512, # Dimension of global observation feature
        use_gated_attention: bool = False,
        gate_mode: str = "element-wise",
        dropout: float = 0.0,
        activation: str = "gelu",
        dim_feedforward: int = 1024,
        mlp_type: str = "standard",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # 1. Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # 2. Positional Embedding (Sinusoidal for better extrapolation)
        # Fixed buffer, not a parameter
        self.register_buffer("pos_embed", self._get_sinusoid_encoding_table(max_seq_len, hidden_size))

        # 3. Conditioning (Timestep + Observation)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.obs_proj = nn.Linear(obs_dim, hidden_size)
        
        # Combine T and Obs: Concat -> MLP is usually better than Sum
        self.cond_combiner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 4. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, 
                num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                use_gated_attention=use_gated_attention,
                gate_mode=gate_mode,
                mlp_type=mlp_type,
            ) for _ in range(depth)
        ])

        # 5. Output Heads
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_size, input_dim)
        
        # Final Zero-Init for output projection
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Standard sinusoidal position encoding table"""
        def get_position_angle_vec(position):
            return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = torch.FloatTensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return sinusoid_table.unsqueeze(0) # [1, L, D]

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        global_cond: Tensor,
    ) -> Tensor:
        """
        x: [B, seq_len, input_dim]
        timestep: [B]
        global_cond: [B, obs_dim]
        """
        B, L, _ = x.shape
        
        # 1. Embed Inputs
        x = self.input_proj(x) # [B, L, H]
        x = x + self.pos_embed[:, :L, :] # Add positional embedding

        # 2. Embed Conditioning
        t_emb = self.t_embedder(timestep) # [B, H]
        obs_emb = self.obs_proj(global_cond) # [B, H]
        
        # Combine: [B, 2*H] -> [B, H]
        # This is safer than sum, allowing model to weigh time vs obs
        c = self.cond_combiner(torch.cat([t_emb, obs_emb], dim=-1)) 

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)

        # 4. Output
        # Re-modulate final norm based on condition (Optional but common in DiT)
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        return x