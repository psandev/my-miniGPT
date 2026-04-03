"""
Prediction head modules for MiniGPT.

Implementations
---------------
* **SingleTokenPrediction (STP)** -- ``norm -> linear -> vocab logits``.
  The output projection weight is tied with the token embedding.
* **MultiTokenPrediction (MTP)** -- Main head (weight-tied, same as STP) plus
  *n-1* auxiliary heads that predict tokens t+2, t+3, ..., t+n.  Each auxiliary
  head: ``linear -> norm -> linear -> vocab``.

  Reference: Gloeckle et al., *Better & Faster Large Language Models via
  Multi-token Prediction*, 2024 (Meta / DeepSeek-V3).

Factory
-------
``build_head(config, embedding_weight)`` returns the appropriate prediction head.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model.config import ModelConfig
from miniGPT.modules.norms import build_norm


class SingleTokenPrediction(nn.Module):
    """Standard next-token prediction head.

    ``logits = linear(norm(hidden))``

    The linear layer's weight is tied to the token embedding matrix.
    """

    def __init__(self, config: ModelConfig, embedding_weight: nn.Parameter) -> None:
        super().__init__()
        self.norm = build_norm(config, config.d_model)
        self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.proj.weight = embedding_weight

    def forward(
        self,
        hidden: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        hidden : Tensor
            Shape ``(B, S, D)``.
        targets : Tensor, optional
            Target token IDs of shape ``(B, S)``.

        Returns
        -------
        dict
            ``{"logits": Tensor, "loss": Tensor | None}``
        """
        logits = self.proj(self.norm(hidden))  # (B, S, V)
        loss = None
        if targets is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}


class MultiTokenPrediction(nn.Module):
    """Multi-token prediction head.

    The main head is identical to STP (weight-tied with the embedding).
    Additionally, *n-1* auxiliary heads predict tokens at positions
    t+2, t+3, ..., t+n.  Each auxiliary head consists of:
    ``projection(d_model -> d_model) -> norm -> output(d_model -> vocab_size)``

    The auxiliary losses are averaged and scaled by ``mtp_loss_weight``.

    Reference: Gloeckle et al., *Better & Faster Large Language Models via
    Multi-token Prediction*, 2024.
    """

    def __init__(self, config: ModelConfig, embedding_weight: nn.Parameter) -> None:
        super().__init__()
        self.n_heads = config.mtp_n_heads
        self.loss_weight = config.mtp_loss_weight
        self.vocab_size = config.vocab_size

        # Main head (weight-tied)
        self.main_norm = build_norm(config, config.d_model)
        self.main_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.main_proj.weight = embedding_weight

        # Auxiliary heads (not weight-tied)
        self.aux_projs = nn.ModuleList()
        self.aux_norms = nn.ModuleList()
        self.aux_outputs = nn.ModuleList()
        for _ in range(self.n_heads - 1):
            self.aux_projs.append(nn.Linear(config.d_model, config.d_model, bias=False))
            self.aux_norms.append(build_norm(config, config.d_model))
            self.aux_outputs.append(nn.Linear(config.d_model, config.vocab_size, bias=False))

    def forward(
        self,
        hidden: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        hidden : Tensor
            Shape ``(B, S, D)``.
        targets : Tensor, optional
            Target token IDs of shape ``(B, S)``.

        Returns
        -------
        dict
            ``{"logits": Tensor, "loss": Tensor | None}``
            where ``logits`` is from the main head.
        """
        # Main head
        main_logits = self.main_proj(self.main_norm(hidden))  # (B, S, V)
        loss = None

        if targets is not None:
            # Main loss: standard next-token
            shift_logits = main_logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

            # Auxiliary losses: predict t+k for k = 2..n_heads
            aux_losses = []
            for k, (proj, norm, out) in enumerate(
                zip(self.aux_projs, self.aux_norms, self.aux_outputs), start=2
            ):
                aux_logits = out(norm(proj(hidden)))  # (B, S, V)
                # Predict token at position t+k: shift by k
                if aux_logits.size(1) > k:
                    aux_shift_logits = aux_logits[:, :-k, :].contiguous()
                    aux_shift_targets = targets[:, k:].contiguous()
                    aux_loss = F.cross_entropy(
                        aux_shift_logits.view(-1, aux_shift_logits.size(-1)),
                        aux_shift_targets.view(-1),
                        ignore_index=-100,
                    )
                    aux_losses.append(aux_loss)

            total_aux = (
                sum(aux_losses) / len(aux_losses) if aux_losses else torch.tensor(0.0, device=hidden.device)
            )
            loss = main_loss + self.loss_weight * total_aux

        return {"logits": main_logits, "loss": loss}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_head(config: ModelConfig, embedding_weight: nn.Parameter) -> nn.Module:
    """Build a prediction head from *config*.

    Parameters
    ----------
    config : ModelConfig
        Must have ``prediction_type`` in ``{"stp", "mtp"}``.
    embedding_weight : nn.Parameter
        Token embedding weight tensor for weight tying.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If ``config.prediction_type`` is not recognised.
    """
    if config.prediction_type == "stp":
        return SingleTokenPrediction(config, embedding_weight)
    if config.prediction_type == "mtp":
        return MultiTokenPrediction(config, embedding_weight)
    raise ValueError(
        f"Unknown prediction_type={config.prediction_type!r}. "
        f"Valid options: 'stp', 'mtp'."
    )
