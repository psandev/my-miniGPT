"""TorchTitan integration for MiniGPT.

Registers MiniGPT as a custom model in TorchTitan's model registry,
enabling FSDP2 distributed training, torch.compile kernel fusion,
bf16 mixed precision, and activation checkpointing.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from configs.model.config import ModelConfig, TrainingConfig, PRESETS
from miniGPT.model import MiniGPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory that TorchTitan calls to instantiate the model
# ---------------------------------------------------------------------------

def build_minigpt_for_torchtitan(
    model_config: dict[str, Any],
) -> MiniGPT:
    """Build a MiniGPT model from a flat dictionary of config values.

    TorchTitan passes the ``[model]`` section of the TOML as a dict.
    We reconstruct a ``ModelConfig`` from it and instantiate the model.
    """
    # Pop the preset key if present; otherwise build from raw fields.
    preset_name = model_config.pop("preset", None)
    if preset_name and preset_name in PRESETS:
        cfg = PRESETS[preset_name]
        # Allow per-field overrides on top of the preset.
        for key, value in model_config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    else:
        cfg = ModelConfig(**model_config)

    model = MiniGPT(cfg)
    param_info = model.count_parameters()
    logger.info(
        "MiniGPT instantiated: %s total parameters (%s)",
        param_info.get("total_human", param_info.get("total", "?")),
        cfg.attention_type,
    )
    return model


# ---------------------------------------------------------------------------
# FSDP2 wrapping policy
# ---------------------------------------------------------------------------

def get_fsdp_wrap_policy(model: MiniGPT) -> partial:
    """Return an FSDP wrapping policy that shards at the TransformerBlock level.

    Each transformer block is wrapped as its own FSDP unit so that
    parameters, gradients, and optimizer states are sharded independently.
    """
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    # Import the block class used by MiniGPT.  The model stores layers in
    # ``model.layers`` which is an nn.ModuleList of block instances.
    block_cls: type = type(model.layers[0]) if len(model.layers) > 0 else nn.Module
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_cls},
    )


# ---------------------------------------------------------------------------
# Activation checkpointing
# ---------------------------------------------------------------------------

def apply_activation_checkpointing(model: MiniGPT) -> None:
    """Enable activation checkpointing on every transformer block.

    This trades compute for memory by recomputing activations during the
    backward pass rather than storing them.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing as _apply_ac,
        checkpoint_wrapper,
        CheckpointImpl,
    )

    block_cls = type(model.layers[0]) if len(model.layers) > 0 else nn.Module

    _apply_ac(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=lambda submodule: isinstance(submodule, block_cls),
    )
    logger.info("Activation checkpointing applied to %s blocks", len(model.layers))


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------

def compile_model(model: MiniGPT, **compile_kwargs: Any) -> MiniGPT:
    """Compile the model with ``torch.compile`` for kernel fusion.

    Keyword arguments are forwarded to ``torch.compile`` (e.g.
    ``mode="reduce-overhead"``).
    """
    compile_kwargs.setdefault("mode", "default")
    compiled = torch.compile(model, **compile_kwargs)
    logger.info("Model compiled with torch.compile (mode=%s)", compile_kwargs["mode"])
    return compiled  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Mixed precision context helper
# ---------------------------------------------------------------------------

def get_autocast_context(device_type: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Return an ``torch.autocast`` context manager for bf16 mixed precision."""
    return torch.autocast(device_type=device_type, dtype=dtype)


# ---------------------------------------------------------------------------
# Registration helper (for TorchTitan plugin system)
# ---------------------------------------------------------------------------

_REGISTERED = False


def register_minigpt() -> None:
    """Register MiniGPT with TorchTitan's model registry.

    This function is idempotent.  It attempts to import ``torchtitan``
    and register the model factory.  If TorchTitan is not installed the
    function logs a warning and returns silently.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from torchtitan.models import model_name_to_cls  # type: ignore[import-untyped]

        model_name_to_cls["minigpt"] = build_minigpt_for_torchtitan
        _REGISTERED = True
        logger.info("MiniGPT registered in TorchTitan model registry")
    except ImportError:
        logger.warning(
            "torchtitan not installed — skipping model registry. "
            "Install with: pip install torchtitan"
        )
    except Exception as exc:  # noqa: BLE001
        # TorchTitan's internal API may change; log and continue.
        logger.warning("Failed to register MiniGPT with TorchTitan: %s", exc)
        # Fallback: try the newer registration API
        try:
            from torchtitan.components.model_converter import register_model  # type: ignore[import-untyped]

            register_model("minigpt", build_minigpt_for_torchtitan)
            _REGISTERED = True
            logger.info("MiniGPT registered via torchtitan register_model()")
        except Exception:
            logger.warning("Could not register via fallback API either")


# ---------------------------------------------------------------------------
# Convenience: set up everything for a training run
# ---------------------------------------------------------------------------

def setup_training(
    config: ModelConfig,
    training_config: TrainingConfig | None = None,
    *,
    use_compile: bool = True,
    use_activation_checkpointing: bool = True,
    device: str = "cuda",
) -> MiniGPT:
    """One-call setup: build model, apply optimizations, move to device.

    Parameters
    ----------
    config:
        Model architecture configuration.
    training_config:
        Optional training hyper-parameters (unused here but reserved for
        future optimizer setup).
    use_compile:
        Whether to ``torch.compile`` the model.
    use_activation_checkpointing:
        Whether to apply gradient checkpointing.
    device:
        Target device string.

    Returns
    -------
    MiniGPT
        The fully-configured model ready for training.
    """
    register_minigpt()

    model = MiniGPT(config)
    model = model.to(device)

    if use_activation_checkpointing:
        try:
            apply_activation_checkpointing(model)
        except Exception as exc:
            logger.warning("Activation checkpointing failed: %s", exc)

    if use_compile and device == "cuda":
        try:
            model = compile_model(model)
        except Exception as exc:
            logger.warning("torch.compile failed: %s", exc)

    param_info = model.count_parameters()
    logger.info("Training setup complete. Parameters: %s", param_info)
    return model
