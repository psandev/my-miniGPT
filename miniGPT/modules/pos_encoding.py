"""
Positional encoding modules for MiniGPT.

Implementations
---------------
* **RoPE** -- Rotary Position Embeddings (Su et al., 2021).  Complex-exponential
  formulation applied directly to Q and K inside the attention layer.
* **Learned** -- Absolute learned positional embeddings via ``nn.Embedding``.
* **ALiBi** -- Attention with Linear Biases (Press et al., 2022).  Position
  information is injected as a linear bias on the attention scores.
* **None** -- No positional encoding (identity passthrough).

Factory
-------
``build_pos_encoding(config)`` returns the appropriate module (or ``None``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from configs.model.config import ModelConfig


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Precomputes the complex-exponential frequency table once and caches it.
    Provides :meth:`apply_rotary` to rotate Q and K tensors.

    Reference: Su et al., *RoFormer: Enhanced Transformer with Rotary Position
    Embedding*, 2021.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 10_000.0,
        scaling: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling = scaling

        # Precompute frequency table as complex exponentials
        freqs = self._build_freqs(head_dim, max_seq_len, base, scaling)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    @staticmethod
    def _build_freqs(
        head_dim: int,
        max_seq_len: int,
        base: float,
        scaling: Optional[float],
    ) -> torch.Tensor:
        """Return complex exponential table of shape ``(max_seq_len, head_dim//2)``."""
        dim = head_dim
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        if scaling is not None:
            t = t / scaling
        freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
        return torch.polar(torch.ones_like(freqs), freqs)  # complex

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to *q* and *k*.

        Parameters
        ----------
        q, k : Tensor
            Shape ``(B, n_heads, S, head_dim)``.
        positions : Tensor, optional
            Integer position indices of shape ``(B, S)`` or ``(S,)``.
            If *None*, assumes contiguous ``[0, S)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Rotated Q and K with the same shape and dtype.
        """
        seq_len = q.size(-2)
        if positions is not None:
            freqs = self.freqs_cis[positions]  # (B, S, dim//2) or (S, dim//2)
        else:
            freqs = self.freqs_cis[:seq_len]  # (S, dim//2)

        # Reshape for broadcast: need (..., S, dim//2)
        while freqs.dim() < q.dim():
            freqs = freqs.unsqueeze(-3)  # add heads dim

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            # x: (B, H, S, D) -> view as complex pairs
            x_shape = x.shape
            x_complex = torch.view_as_complex(
                x.float().reshape(*x_shape[:-1], -1, 2)
            )  # (..., D//2)
            x_rotated = x_complex * freqs
            return torch.view_as_real(x_rotated).reshape(x_shape).type_as(x)

        return _rotate(q), _rotate(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity -- RoPE is applied inside attention via :meth:`apply_rotary`."""
        return x


class LearnedPositionEncoding(nn.Module):
    """Absolute learned positional embeddings.

    An ``nn.Embedding`` table of shape ``(max_seq_len, d_model)`` added to the
    token embeddings before the first Transformer block.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to *x* of shape ``(B, S, D)``."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)


class ALiBiPositionEncoding(nn.Module):
    """Attention with Linear Biases (ALiBi).

    Computes a causal linear bias matrix added to the attention scores.  The
    bias slopes are fixed per head (geometric series).

    Reference: Press et al., *Train Short, Test Long: Attention with Linear
    Biases Enables Input Length Extrapolation*, ICLR 2022.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.n_heads = n_heads
        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", slopes, persistent=False)  # (n_heads,)
        # Precompute bias for max length; we'll slice at runtime
        bias = self._build_bias(max_seq_len, slopes)
        self.register_buffer("bias", bias, persistent=False)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        """Geometric head slopes as described in the ALiBi paper."""

        def _slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(_slopes_power_of_2(n_heads), dtype=torch.float32)

        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _slopes_power_of_2(closest_pow2)
        extra = _slopes_power_of_2(2 * closest_pow2)
        slopes.extend(extra[0::2][: n_heads - closest_pow2])
        return torch.tensor(slopes, dtype=torch.float32)

    @staticmethod
    def _build_bias(seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
        """Build the ALiBi bias matrix of shape ``(1, n_heads, seq_len, seq_len)``."""
        positions = torch.arange(seq_len)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)  # (S, S)
        relative = relative.float()
        # Causal: mask future positions with large negative
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        relative.masked_fill_(causal_mask, 0)
        # Only keep non-positive distances (causal)
        relative = -relative.abs()
        bias = slopes.unsqueeze(1).unsqueeze(1) * relative.unsqueeze(0)  # (H, S, S)
        return bias.unsqueeze(0)  # (1, H, S, S)

    def get_bias(self, seq_len: int) -> torch.Tensor:
        """Return the ALiBi bias for a given sequence length.

        Returns shape ``(1, n_heads, seq_len, seq_len)``.
        """
        if seq_len <= self.bias.size(-1):
            return self.bias[:, :, :seq_len, :seq_len]
        # Recompute for longer sequences
        return self._build_bias(seq_len, self.slopes).to(self.bias.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity -- ALiBi bias is applied inside attention via :meth:`get_bias`."""
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pos_encoding(config: ModelConfig) -> nn.Module | None:
    """Build a positional encoding module from *config*.

    Parameters
    ----------
    config : ModelConfig
        Must have ``pos_encoding`` in ``{"rope", "learned", "alibi", "none"}``.

    Returns
    -------
    nn.Module or None
        ``None`` when ``pos_encoding == "none"``.

    Raises
    ------
    ValueError
        If ``config.pos_encoding`` is not recognised.
    """
    if config.pos_encoding == "rope":
        return RotaryPositionEncoding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
            scaling=config.rope_scaling,
        )
    if config.pos_encoding == "learned":
        return LearnedPositionEncoding(config.max_seq_len, config.d_model)
    if config.pos_encoding == "alibi":
        return ALiBiPositionEncoding(config.n_heads, config.max_seq_len)
    if config.pos_encoding == "none":
        return None
    raise ValueError(
        f"Unknown pos_encoding={config.pos_encoding!r}. "
        f"Valid options: 'rope', 'learned', 'alibi', 'none'."
    )
