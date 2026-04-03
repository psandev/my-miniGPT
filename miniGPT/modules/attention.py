"""
Attention mechanisms for MiniGPT.

Implementations
---------------
* **GroupedQueryAttention** -- Unified MHA / GQA / MQA attention.
  ``n_kv_heads == n_heads`` gives standard Multi-Head Attention (Vaswani et al., 2017),
  ``n_kv_heads < n_heads`` gives Grouped-Query Attention (Ainslie et al., 2023),
  ``n_kv_heads == 1`` gives Multi-Query Attention (Shazeer, 2019).
  Uses ``F.scaled_dot_product_attention`` for automatic Flash Attention dispatch.

* **MLAAttention** -- Multi-head Latent Attention with low-rank KV compression
  and decoupled RoPE (DeepSeek-V2, Dai et al., 2024).

Factory
-------
``build_attention(config)`` returns the appropriate attention module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model.config import ModelConfig
from miniGPT.modules.pos_encoding import RotaryPositionEncoding, ALiBiPositionEncoding


class GroupedQueryAttention(nn.Module):
    """Unified Multi-Head / Grouped-Query / Multi-Query Attention.

    Projection shapes adapt based on ``n_kv_heads``:
    * ``n_kv_heads == n_heads`` -- standard MHA.
    * ``1 < n_kv_heads < n_heads`` -- GQA (Ainslie et al., 2023).
    * ``n_kv_heads == 1`` -- MQA (Shazeer, 2019).

    All variants delegate the dot-product to
    ``F.scaled_dot_product_attention`` which automatically dispatches to
    Flash Attention 2 / Memory-Efficient Attention when available.

    No bias in Q / K / V / O projections (modern default).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.n_groups = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_encoder: nn.Module | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, S, D)``.
        pos_encoder : module, optional
            RoPE or ALiBi encoder.
        attention_mask : Tensor, optional
            Boolean mask of shape ``(B, 1, S, S)`` or ``(B, 1, 1, S)``.
            ``True`` positions are **masked** (ignored).
        kv_cache : tuple, optional
            Previous ``(k, v)`` tensors for incremental decoding.
        positions : Tensor, optional
            Position indices for RoPE.

        Returns
        -------
        (output, new_kv_cache)
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if isinstance(pos_encoder, RotaryPositionEncoding):
            q, k = pos_encoder.apply_rotary(q, k, positions)

        # KV cache for autoregressive decoding
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads for GQA: repeat each KV head to match query groups
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Build attention mask for SDPA
        attn_mask = None
        if isinstance(pos_encoder, ALiBiPositionEncoding):
            # ALiBi: additive bias; SDPA interprets float mask as additive
            alibi_bias = pos_encoder.get_bias(k.size(2)).to(dtype=q.dtype, device=q.device)
            # Slice for query positions (handles KV cache)
            alibi_bias = alibi_bias[:, :, -S:, :k.size(2)]
            attn_mask = alibi_bias
            if attention_mask is not None:
                attn_mask = attn_mask.masked_fill(attention_mask[:, :, -S:, :k.size(2)], float("-inf"))
        elif attention_mask is not None:
            # Convert boolean mask to float for SDPA
            attn_mask = torch.zeros(B, 1, S, k.size(2), dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(attention_mask[:, :, -S:, :k.size(2)], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attn_mask is None),  # use causal mask only if no explicit mask
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_kv_cache


class MLAAttention(nn.Module):
    """Multi-head Latent Attention (MLA).

    Low-rank KV compression with decoupled RoPE, as introduced in DeepSeek-V2
    (Dai et al., 2024).

    The input is projected into a low-rank latent ``c_kv`` of dimension
    ``kv_lora_rank``, from which K and V are decompressed per head.  A separate
    small ``rope_head_dim`` portion of the key is computed with its own
    projection and receives RoPE independently, enabling decoupled position
    encoding.

    No bias in any projection.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.kv_lora_rank = config.mla_kv_lora_rank
        self.q_lora_rank = config.mla_q_lora_rank
        self.rope_head_dim = config.mla_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.dropout = config.dropout

        # Q path: compress → decompress + separate RoPE portion
        self.q_down = nn.Linear(config.d_model, self.q_lora_rank, bias=False)
        self.q_up = nn.Linear(
            self.q_lora_rank, self.n_heads * self.nope_head_dim, bias=False
        )
        self.q_rope = nn.Linear(
            self.q_lora_rank, self.n_heads * self.rope_head_dim, bias=False
        )

        # KV path: compress to latent, decompress K and V per head
        self.kv_down = nn.Linear(config.d_model, self.kv_lora_rank, bias=False)
        self.k_up = nn.Linear(
            self.kv_lora_rank, self.n_heads * self.nope_head_dim, bias=False
        )
        self.k_rope = nn.Linear(config.d_model, self.n_heads * self.rope_head_dim, bias=False)
        self.v_up = nn.Linear(
            self.kv_lora_rank, self.n_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        # Decoupled RoPE for MLA
        self.rope = RotaryPositionEncoding(
            head_dim=self.rope_head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            scaling=config.rope_scaling,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_encoder: nn.Module | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape

        # Q: compress → decompress non-RoPE + RoPE portions
        q_latent = self.q_down(x)
        q_nope = self.q_up(q_latent).view(B, S, self.n_heads, self.nope_head_dim).transpose(1, 2)
        q_pe = self.q_rope(q_latent).view(B, S, self.n_heads, self.rope_head_dim).transpose(1, 2)

        # KV: compress → decompress
        kv_latent = self.kv_down(x)
        k_nope = self.k_up(kv_latent).view(B, S, self.n_heads, self.nope_head_dim).transpose(1, 2)
        k_pe = self.k_rope(x).view(B, S, self.n_heads, self.rope_head_dim).transpose(1, 2)
        v = self.v_up(kv_latent).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply decoupled RoPE to the RoPE portions
        q_pe, k_pe = self.rope.apply_rotary(q_pe, k_pe, positions)

        # Concatenate non-RoPE and RoPE parts for Q and K
        q = torch.cat([q_nope, q_pe], dim=-1)  # (B, H, S, head_dim)
        k = torch.cat([k_nope, k_pe], dim=-1)  # (B, H, S, head_dim)

        # KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Attention mask
        attn_mask = None
        if attention_mask is not None:
            attn_mask = torch.zeros(B, 1, S, k.size(2), dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(attention_mask[:, :, -S:, :k.size(2)], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attn_mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_kv_cache


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_attention(config: ModelConfig) -> nn.Module:
    """Build an attention module from *config*.

    Parameters
    ----------
    config : ModelConfig
        Must have ``attention_type`` in ``{"mha", "gqa", "mla"}``.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If ``config.attention_type`` is not recognised.
    """
    if config.attention_type in ("mha", "gqa"):
        return GroupedQueryAttention(config)
    if config.attention_type == "mla":
        return MLAAttention(config)
    raise ValueError(
        f"Unknown attention_type={config.attention_type!r}. "
        f"Valid options: 'mha', 'gqa', 'mla'."
    )
