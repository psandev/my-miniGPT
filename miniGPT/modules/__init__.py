"""
Swappable Transformer components for MiniGPT.

All modules are accessed through factory functions that read a
:class:`~configs.model.config.ModelConfig` and return the appropriate
implementation.

Factories
---------
- :func:`build_attention` -- MHA / GQA / MLA
- :func:`build_ffn` -- SwiGLU / GELU / ReLU / MoE
- :func:`build_norm` -- RMSNorm / LayerNorm / DynamicTanh
- :func:`build_pos_encoding` -- RoPE / Learned / ALiBi / None
- :func:`build_residual` -- Standard / mHC
- :func:`build_head` -- STP / MTP
"""

from miniGPT.modules.attention import build_attention
from miniGPT.modules.ffn import build_ffn
from miniGPT.modules.norms import build_norm
from miniGPT.modules.pos_encoding import build_pos_encoding
from miniGPT.modules.residual import build_residual
from miniGPT.modules.prediction import build_head

__all__ = [
    "build_attention",
    "build_ffn",
    "build_norm",
    "build_pos_encoding",
    "build_residual",
    "build_head",
]
