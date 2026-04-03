"""
Residual connection modules for MiniGPT.

Implementations
---------------
* **StandardResidual** -- Classic pre-norm residual: ``x + sublayer(norm(x))``.
  The norm is *passed in* at call time, not created inside.
* **ManifoldHyperConnection (mHC)** -- Maintains *n* parallel residual streams
  mixed through a matrix constrained to the Birkhoff polytope (doubly
  stochastic) via the Sinkhorn-Knopp algorithm.

  Reference: Godfrey et al., *Hyper-Connections*, 2024.

Factory
-------
``build_residual(config)`` returns the appropriate residual module.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from configs.model.config import ModelConfig


class StandardResidual(nn.Module):
    """Standard pre-norm residual connection.

    ``output = x + sublayer(norm(x))``

    Both the sublayer function and norm module are passed into ``forward``
    so that this class is stateless with respect to those components.
    """

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor | tuple],
        norm: nn.Module,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Hidden state of shape ``(B, S, D)``.
        sublayer : callable
            Function (e.g. attention or FFN) to apply to the normalised input.
            May return a tuple; only the first element is used for the residual.
        norm : nn.Module
            Normalisation applied before the sublayer (pre-norm).

        Returns
        -------
        Tensor
            ``x + sublayer_output``, same shape as *x*.
        """
        normed = norm(x)
        out = sublayer(normed)
        if isinstance(out, tuple):
            out = out[0]
        return x + out

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Identity -- standard residual does not use multiple streams."""
        return x

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Identity -- standard residual does not use multiple streams."""
        return x


class ManifoldHyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connections (mHC).

    Maintains ``n_streams`` parallel residual streams.  Before each sublayer
    the streams are mixed through a doubly-stochastic matrix (Birkhoff polytope)
    computed via Sinkhorn-Knopp normalisation, then collapsed to a single stream
    for the sublayer, and re-expanded afterward.

    The mixing matrix ``M`` is parameterised as ``sigmoid(W)`` and then
    projected onto the Birkhoff polytope via alternating row/column
    normalisation (Sinkhorn-Knopp).

    Reference: Godfrey et al., *Hyper-Connections*, 2024.
    """

    def __init__(self, d_model: int, n_streams: int = 4, sinkhorn_iters: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Raw mixing weights (before Sinkhorn projection)
        self.mix_weight = nn.Parameter(torch.randn(n_streams, n_streams) * 0.01)

        # Projection to expand from 1 stream (D) to n streams (N*D)
        self.expand_proj = nn.Linear(d_model, n_streams * d_model, bias=False)
        # Projection to collapse from n streams back to 1
        self.collapse_proj = nn.Linear(n_streams * d_model, d_model, bias=False)

    def _sinkhorn(self, W: torch.Tensor) -> torch.Tensor:
        """Project a non-negative matrix onto the Birkhoff polytope.

        Alternating row and column normalisation (Sinkhorn-Knopp algorithm).
        """
        M = torch.sigmoid(W)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Expand a single-stream hidden state to *n_streams*.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, S, D)``.

        Returns
        -------
        Tensor
            Shape ``(B, S, N, D)`` where ``N = n_streams``.
        """
        B, S, D = x.shape
        expanded = self.expand_proj(x)  # (B, S, N*D)
        return expanded.view(B, S, self.n_streams, D)

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse *n_streams* back to a single stream.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, S, N, D)``.

        Returns
        -------
        Tensor
            Shape ``(B, S, D)``.
        """
        B, S, N, D = x.shape
        flat = x.reshape(B, S, N * D)
        return self.collapse_proj(flat)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor | tuple],
        norm: nn.Module,
    ) -> torch.Tensor:
        """Full pre-norm + residual with stream mixing.

        Parameters
        ----------
        x : Tensor
            Multi-stream hidden state of shape ``(B, S, N, D)``.
        sublayer : callable
            Sublayer function (attention or FFN) expecting ``(B, S, D)`` input.
        norm : nn.Module
            Pre-norm module.

        Returns
        -------
        Tensor
            Updated multi-stream hidden state of shape ``(B, S, N, D)``.
        """
        B, S, N, D = x.shape

        # Compute doubly stochastic mixing matrix
        M = self._sinkhorn(self.mix_weight)  # (N, N)

        # Mix streams: new_i = sum_j M[i,j] * stream_j
        # x: (B, S, N, D), M: (N, N) -> mixed: (B, S, N, D)
        mixed = torch.einsum("ij, bsjd -> bsid", M, x)

        # Collapse to single stream for sublayer: mean over streams
        collapsed = mixed.mean(dim=2)  # (B, S, D)

        # Apply pre-norm + sublayer
        normed = norm(collapsed)
        sub_out = sublayer(normed)
        if isinstance(sub_out, tuple):
            sub_out = sub_out[0]

        # Add sublayer output back to each stream (broadcast)
        residual = sub_out.unsqueeze(2)  # (B, S, 1, D)
        return mixed + residual


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_residual(config: ModelConfig) -> nn.Module:
    """Build a residual connection module from *config*.

    Parameters
    ----------
    config : ModelConfig
        Must have ``residual_type`` in ``{"standard", "mhc"}``.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If ``config.residual_type`` is not recognised.
    """
    if config.residual_type == "standard":
        return StandardResidual()
    if config.residual_type == "mhc":
        return ManifoldHyperConnection(
            d_model=config.d_model,
            n_streams=config.mhc_n_streams,
            sinkhorn_iters=config.mhc_sinkhorn_iters,
        )
    raise ValueError(
        f"Unknown residual_type={config.residual_type!r}. "
        f"Valid options: 'standard', 'mhc'."
    )
