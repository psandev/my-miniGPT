"""
Feed-forward network modules for MiniGPT.

Implementations
---------------
* **SwiGLUFFN** -- Gated FFN with SiLU (Swish) activation and three linear
  layers (Shazeer, 2020; used in LLaMA / Mistral / Gemma).  ``d_ff`` is
  reduced by 2/3 relative to a standard 4x expansion so that total parameter
  count matches.
* **GELUFFN** -- Classic two-layer FFN with GELU activation (Hendrycks & Gimpel, 2016).
* **ReLUFFN** -- Classic two-layer FFN with ReLU activation.
* **MoEFFN** -- Mixture-of-Experts with top-k routing over *n* SwiGLU experts,
  plus a load-balancing auxiliary loss (Shazeer et al., 2017; Lepikhin et al., 2020).

Factory
-------
``build_ffn(config)`` returns the appropriate FFN module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """Gated FFN with SiLU activation.

    ``down(swish(gate(x)) * up(x))``

    Three weight matrices: gate, up (both ``d_model -> d_ff``), and down
    (``d_ff -> d_model``).

    Reference: Shazeer, *GLU Variants Improve Transformer*, 2020.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GELUFFN(nn.Module):
    """Standard two-layer FFN with GELU activation.

    ``down(gelu(up(x)))``

    Reference: Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, 2016.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class ReLUFFN(nn.Module):
    """Standard two-layer FFN with ReLU activation.

    ``down(relu(up(x)))``
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)))


class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN with top-k routing.

    Each expert is a SwiGLU FFN.  A learned gating network routes each token
    to the top-k experts.  A load-balancing auxiliary loss encourages uniform
    expert utilisation.

    Note: This implementation uses a simple Python loop over experts.
    Production systems would use Megablocks or Scattermoe for GPU-efficient
    batched dispatch.

    References
    ----------
    - Shazeer et al., *Outrageously Large Neural Networks: The
      Sparsely-Gated Mixture-of-Experts Layer*, 2017.
    - Lepikhin et al., *GShard: Scaling Giant Models with Conditional
      Computation and Automatic Sharding*, 2020.
    - Fedus et al., *Switch Transformers*, 2021.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFFN(d_model, d_ff, bias=bias) for _ in range(num_experts)]
        )
        # Updated each forward pass; collected by the model for the total loss.
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(B, S, D)``.

        Returns
        -------
        Tensor
            Same shape as input.  Also sets ``self.aux_loss``.
        """
        orig_shape = x.shape
        x_flat = x.view(-1, x.size(-1))  # (B*S, D)
        num_tokens = x_flat.size(0)

        # Router logits → probabilities
        logits = self.gate(x_flat)  # (B*S, E)
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)  # (B*S, k)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # renormalize

        # Compute auxiliary load-balancing loss
        # f_i = fraction of tokens routed to expert i
        # P_i = mean probability assigned to expert i
        # aux_loss = E * sum(f_i * P_i)
        one_hot = F.one_hot(top_k_indices, self.num_experts).float().sum(dim=1)  # (B*S, E)
        f = one_hot.mean(dim=0)  # (E,)
        p = probs.mean(dim=0)  # (E,)
        self.aux_loss = self.aux_loss_weight * self.num_experts * (f * p).sum()

        # Expert computation (simple loop)
        output = torch.zeros_like(x_flat)
        for k_idx in range(self.top_k):
            expert_indices = top_k_indices[:, k_idx]  # (B*S,)
            weights = top_k_probs[:, k_idx]  # (B*S,)
            for e_idx in range(self.num_experts):
                mask = expert_indices == e_idx
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output

        return output.view(orig_shape)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ffn(config: ModelConfig) -> nn.Module:
    """Build a feed-forward module from *config*.

    Parameters
    ----------
    config : ModelConfig
        Must have ``ffn_type`` in ``{"swiglu", "gelu", "relu", "moe"}``.

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If ``config.ffn_type`` is not recognised.
    """
    d_ff = config.d_ff

    if config.ffn_type == "swiglu":
        return SwiGLUFFN(config.d_model, d_ff, bias=config.use_bias)
    if config.ffn_type == "gelu":
        # Standard 4x expansion for non-gated FFN
        d_ff_standard = config.d_model * 4
        return GELUFFN(config.d_model, d_ff_standard, bias=config.use_bias)
    if config.ffn_type == "relu":
        d_ff_standard = config.d_model * 4
        return ReLUFFN(config.d_model, d_ff_standard, bias=config.use_bias)
    if config.ffn_type == "moe":
        return MoEFFN(
            d_model=config.d_model,
            d_ff=d_ff,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            aux_loss_weight=config.moe_aux_loss_weight,
            bias=config.use_bias,
        )
    raise ValueError(
        f"Unknown ffn_type={config.ffn_type!r}. "
        f"Valid options: 'swiglu', 'gelu', 'relu', 'moe'."
    )
