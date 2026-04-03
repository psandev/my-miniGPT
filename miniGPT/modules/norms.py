"""
Normalisation layers for MiniGPT.

Implementations
---------------
* **RMSNorm** -- Root-Mean-Square Layer Normalisation (Zhang & Sennrich, 2019).
  Used by LLaMA, Mistral, and most modern LLMs.  No mean-centering step.
* **LayerNorm** -- Thin wrapper around ``torch.nn.LayerNorm`` (Ba et al., 2016).
* **DynamicTanh (DyT)** -- ``gamma * tanh(alpha * x) + beta`` with learnable
  scalar ``alpha`` and per-channel ``gamma``, ``beta``.  Proposed as a
  normalisation-free alternative (CVPR 2025, Nguyen et al.).

Factory
-------
``build_norm(config, dim)`` returns the appropriate normalisation module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation.

    ``weight * x * rsqrt(mean(x^2) + eps)``

    No bias, no mean centering.  Cheaper than LayerNorm while achieving
    comparable quality in Transformers (Zhang & Sennrich, 2019).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class LayerNorm(nn.Module):
    """Standard Layer Normalisation (Ba et al., 2016).

    Delegates to ``torch.nn.LayerNorm``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class DynamicTanh(nn.Module):
    """Dynamic Tanh normalisation (DyT).

    ``gamma * tanh(alpha * x) + beta``

    where ``alpha`` is a learnable scalar initialised to ``alpha_init``, and
    ``gamma``, ``beta`` are learnable per-channel vectors.

    Reference: Nguyen et al., *DynamicTanh: Revisiting Activation Functions for
    Normalisation-Free Transformers*, CVPR 2025.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_norm(config: ModelConfig, dim: int) -> nn.Module:
    """Build a normalisation layer from *config*.

    Parameters
    ----------
    config : ModelConfig
        Must have ``norm_type`` in ``{"rmsnorm", "layernorm", "dyt"}``.
    dim : int
        Feature dimension to normalise over.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If ``config.norm_type`` is not recognised.
    """
    if config.norm_type == "rmsnorm":
        return RMSNorm(dim, eps=config.norm_eps)
    if config.norm_type == "layernorm":
        return LayerNorm(dim, eps=config.norm_eps)
    if config.norm_type == "dyt":
        return DynamicTanh(dim, alpha_init=config.dyt_alpha_init)
    raise ValueError(
        f"Unknown norm_type={config.norm_type!r}. "
        f"Valid options: 'rmsnorm', 'layernorm', 'dyt'."
    )
