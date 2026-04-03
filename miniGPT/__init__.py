"""
MiniGPT -- A modular, research-oriented decoder-only Transformer framework.

Exports
-------
- :class:`MiniGPT` -- The main model class.
- :class:`ModelConfig` -- Configuration dataclass.
- :data:`PRESETS` -- Pre-defined model configurations.
- :func:`generate` -- Autoregressive text generation with KV cache.
"""

from configs.model.config import ModelConfig, PRESETS
from miniGPT.model import MiniGPT
from miniGPT.generation import generate

__all__ = ["MiniGPT", "ModelConfig", "PRESETS", "generate"]
