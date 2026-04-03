"""
MiniGPT -- full decoder-only Transformer model.

Assembles all swappable components (attention, FFN, norm, positional encoding,
residual connection, prediction head) from a single
:class:`~configs.model.config.ModelConfig`.

Architecture (pre-norm):

.. code-block:: text

    Input IDs
      -> Token Embedding (+ Learned PosEnc if applicable)
      -> [mHC: expand to n streams]
      -> Transformer Block x N:
          -> Norm -> Attention (+ RoPE/ALiBi) -> Residual
          -> Norm -> FFN -> Residual
      -> [mHC: collapse streams]
      -> Prediction Head (STP or MTP)
      -> Logits + Loss

Weight tying between token embedding and main output projection.
MoE auxiliary losses are collected from all layers and summed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from configs.model.config import ModelConfig
from miniGPT.modules.attention import build_attention, GroupedQueryAttention, MLAAttention
from miniGPT.modules.ffn import build_ffn, MoEFFN
from miniGPT.modules.norms import build_norm
from miniGPT.modules.pos_encoding import build_pos_encoding, LearnedPositionEncoding
from miniGPT.modules.residual import build_residual, ManifoldHyperConnection
from miniGPT.modules.prediction import build_head


class TransformerBlock(nn.Module):
    """Single Transformer decoder block.

    Pre-norm architecture: ``norm -> sublayer -> residual`` for both
    attention and FFN sub-layers.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.attn = build_attention(config)
        self.ffn = build_ffn(config)
        self.attn_norm = build_norm(config, config.d_model)
        self.ffn_norm = build_norm(config, config.d_model)
        self.attn_residual = build_residual(config)
        self.ffn_residual = build_residual(config)

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
            Hidden state.  Shape ``(B, S, D)`` for standard residual or
            ``(B, S, N, D)`` for mHC.
        pos_encoder : nn.Module | None
            Positional encoding module (RoPE, ALiBi, or None).
        attention_mask : Tensor | None
            Boolean causal + padding mask.
        kv_cache : tuple | None
            Previous (k, v) for incremental decoding.
        positions : Tensor | None
            Position indices for RoPE.

        Returns
        -------
        (hidden, new_kv_cache)
        """
        new_kv = None

        # -- Attention sub-layer --
        if isinstance(self.attn_residual, ManifoldHyperConnection):
            # mHC: x is (B, S, N, D); sublayer expects (B, S, D)
            def attn_fn(h: torch.Tensor) -> torch.Tensor:
                nonlocal new_kv
                out, new_kv = self.attn(
                    h, pos_encoder=pos_encoder, attention_mask=attention_mask,
                    kv_cache=kv_cache, positions=positions,
                )
                return out
            x = self.attn_residual(x, attn_fn, self.attn_norm)
        else:
            normed = self.attn_norm(x)
            attn_out, new_kv = self.attn(
                normed, pos_encoder=pos_encoder, attention_mask=attention_mask,
                kv_cache=kv_cache, positions=positions,
            )
            x = x + attn_out

        # -- FFN sub-layer --
        if isinstance(self.ffn_residual, ManifoldHyperConnection):
            x = self.ffn_residual(x, self.ffn, self.ffn_norm)
        else:
            normed = self.ffn_norm(x)
            x = x + self.ffn(normed)

        return x, new_kv


class MiniGPT(nn.Module):
    """MiniGPT: modular decoder-only Transformer.

    All architectural components are determined by *config* and built
    through factory functions, making every component swappable via a
    single configuration change.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding
        self.pos_encoder = build_pos_encoding(config)

        # Residual type (we need to know if mHC for expand/collapse)
        self.use_mhc = config.residual_type == "mhc"
        if self.use_mhc:
            # Shared expand/collapse across layers is handled per-block,
            # but we need one expand at start and one collapse at end.
            self.stream_expand = ManifoldHyperConnection(
                d_model=config.d_model,
                n_streams=config.mhc_n_streams,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
            )
            # For final collapse, reuse the collapse_proj from stream_expand
            # or use a separate one. We'll use the same module's collapse method.

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.n_layers)]
        )

        # Dropout (if any)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Prediction head (weight-tied)
        self.head = build_head(config, self.tok_emb.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Full forward pass.

        Parameters
        ----------
        input_ids : Tensor
            Token IDs of shape ``(B, S)``.
        targets : Tensor, optional
            Target token IDs of shape ``(B, S)`` for loss computation.
        attention_mask : Tensor, optional
            Boolean mask where ``True`` positions are **masked** (ignored).
            Shape ``(B, 1, S, S)`` or broadcastable.

        Returns
        -------
        dict
            ``{"logits": Tensor, "loss": Tensor | None, "aux_loss": Tensor | None}``
        """
        B, S = input_ids.shape

        # 1. Token embedding
        x = self.tok_emb(input_ids)  # (B, S, D)

        # 2. Add learned positional encoding if applicable
        if isinstance(self.pos_encoder, LearnedPositionEncoding):
            x = self.pos_encoder(x)

        x = self.dropout(x)

        # 3. Build causal mask if none provided
        if attention_mask is None:
            # Standard causal mask: True = masked
            causal = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1
            )
            attention_mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # 4. mHC: expand to n streams
        if self.use_mhc:
            x = self.stream_expand.expand(x)  # (B, S, N, D)

        # 5. Transformer blocks
        for layer in self.layers:
            x, _ = layer(
                x,
                pos_encoder=self.pos_encoder,
                attention_mask=attention_mask,
            )

        # 6. mHC: collapse streams
        if self.use_mhc:
            x = self.stream_expand.collapse(x)  # (B, S, D)

        # 7. Prediction head
        head_out = self.head(x, targets=targets)

        # 8. Collect MoE auxiliary losses
        aux_loss = None
        moe_losses = []
        for layer in self.layers:
            if isinstance(layer.ffn, MoEFFN) and layer.ffn.aux_loss is not None:
                moe_losses.append(layer.ffn.aux_loss)
        if moe_losses:
            aux_loss = sum(moe_losses)

        # Combine losses
        total_loss = head_out["loss"]
        if total_loss is not None and aux_loss is not None:
            total_loss = total_loss + aux_loss

        return {
            "logits": head_out["logits"],
            "loss": total_loss,
            "aux_loss": aux_loss,
        }

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with KV cache for autoregressive generation.

        Parameters
        ----------
        input_ids : Tensor
            Token IDs of shape ``(B, S)`` -- typically ``S=1`` during generation.
        kv_caches : dict, optional
            Mapping from layer index to ``(k_cache, v_cache)`` tensors.
        positions : Tensor, optional
            Position indices of shape ``(B, S)`` or ``(S,)``.
        attention_mask : Tensor, optional
            Attention mask.

        Returns
        -------
        (logits, new_kv_caches)
        """
        if kv_caches is None:
            kv_caches = {}
        new_kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        B, S = input_ids.shape
        x = self.tok_emb(input_ids)

        if isinstance(self.pos_encoder, LearnedPositionEncoding):
            x = self.pos_encoder(x)

        # For cached generation we skip mHC to keep things simple and
        # compatible with standard serving (per spec: mHC collapsed at export)
        for i, layer in enumerate(self.layers):
            cache = kv_caches.get(i)
            # For generation, use the standard (non-mHC) path even if model
            # was trained with mHC, since the hidden is (B,S,D)
            normed = layer.attn_norm(x)
            attn_out, new_kv = layer.attn(
                normed,
                pos_encoder=self.pos_encoder,
                attention_mask=attention_mask,
                kv_cache=cache,
                positions=positions,
            )
            x = x + attn_out

            normed = layer.ffn_norm(x)
            x = x + layer.ffn(normed)

            new_kv_caches[i] = new_kv

        # Final norm + projection (just the main head norm + proj)
        x = self.head.norm(x) if hasattr(self.head, 'norm') else self.head.main_norm(x)
        proj = self.head.proj if hasattr(self.head, 'proj') else self.head.main_proj
        logits = proj(x)

        return logits, new_kv_caches

    def count_parameters(self) -> dict[str, str | int]:
        """Parameter count broken down by component.

        Returns
        -------
        dict
            Keys: ``embedding``, ``attention``, ``ffn``, ``norms``, ``head``,
            ``residual``, ``total``, plus ``*_human`` variants with M/B suffix.
        """
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        def _human(n: int) -> str:
            if n >= 1e9:
                return f"{n / 1e9:.2f}B"
            if n >= 1e6:
                return f"{n / 1e6:.2f}M"
            if n >= 1e3:
                return f"{n / 1e3:.1f}K"
            return str(n)

        emb = _count(self.tok_emb)
        attn = sum(_count(l.attn) for l in self.layers)
        ffn = sum(_count(l.ffn) for l in self.layers)
        norms = sum(
            _count(l.attn_norm) + _count(l.ffn_norm) for l in self.layers
        )
        head = sum(
            p.numel() for p in self.head.parameters()
            if p.data_ptr() != self.tok_emb.weight.data_ptr()
        )
        residual = sum(
            _count(l.attn_residual) + _count(l.ffn_residual) for l in self.layers
        )
        if self.use_mhc:
            residual += _count(self.stream_expand)

        total = self.get_num_params()

        return {
            "embedding": emb,
            "embedding_human": _human(emb),
            "attention": attn,
            "attention_human": _human(attn),
            "ffn": ffn,
            "ffn_human": _human(ffn),
            "norms": norms,
            "norms_human": _human(norms),
            "head": head,
            "head_human": _human(head),
            "residual": residual,
            "residual_human": _human(residual),
            "total": total,
            "total_human": _human(total),
        }

    def get_num_params(self) -> int:
        """Total number of parameters (unique, accounts for weight tying)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            pid = p.data_ptr()
            if pid not in seen:
                seen.add(pid)
                total += p.numel()
        return total
