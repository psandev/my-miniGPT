"""VRAM estimator for MiniGPT training configurations.

Computes memory requirements for model parameters, optimizer states,
activations, and gradients. Reports whether the configuration fits on
24GB or 48GB GPUs and recommends batch size.

Usage::

    python automation/memory_estimator.py --preset medium
    python automation/memory_estimator.py --preset large --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig, TrainingConfig, PRESETS

logger = logging.getLogger(__name__)

# Bytes per parameter for different dtypes
BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
}


def estimate_model_params(config: ModelConfig) -> dict[str, int]:
    """Estimate parameter counts broken down by component.

    Parameters
    ----------
    config:
        Model architecture configuration.

    Returns
    -------
    dict[str, int]
        Parameter counts by component and total.
    """
    d = config.d_model
    d_ff = config.d_ff
    n_layers = config.n_layers
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads or n_heads
    head_dim = config.head_dim
    vocab_size = config.vocab_size

    # Embedding
    embedding_params = vocab_size * d

    # Per-layer attention
    if config.attention_type == "mla":
        # MLA: compressed projections
        attn_per_layer = (
            d * config.mla_q_lora_rank  # down-proj Q
            + config.mla_q_lora_rank * (n_heads * head_dim)  # up-proj Q
            + d * config.mla_kv_lora_rank  # down-proj KV
            + config.mla_kv_lora_rank * (n_kv_heads * head_dim * 2)  # up-proj K, V
            + n_heads * head_dim * d  # output projection
        )
    else:
        # MHA/GQA
        attn_per_layer = (
            d * (n_heads * head_dim)        # Q projection
            + d * (n_kv_heads * head_dim)    # K projection
            + d * (n_kv_heads * head_dim)    # V projection
            + (n_heads * head_dim) * d       # Output projection
        )

    # Per-layer FFN
    if config.ffn_type == "swiglu":
        ffn_per_layer = 3 * d * d_ff  # gate + up + down
    elif config.ffn_type == "moe":
        single_expert = 3 * d * d_ff  # SwiGLU per expert
        ffn_per_layer = (
            config.moe_num_experts * single_expert  # All experts
            + d * config.moe_num_experts             # Router
        )
    else:  # gelu, relu
        ffn_per_layer = 2 * d * d_ff  # up + down

    # Per-layer norms (2 per layer: attn_norm + ffn_norm)
    if config.norm_type == "dyt":
        norm_per_layer = 2 * (d + d + 1)  # gamma + beta + alpha per norm
    elif config.norm_type == "layernorm":
        norm_per_layer = 2 * (d + d)  # weight + bias per norm
    else:  # rmsnorm
        norm_per_layer = 2 * d  # weight only

    # Per-layer residual (mHC)
    residual_per_layer = 0
    if config.residual_type == "mhc":
        n_streams = config.mhc_n_streams
        # Mixing matrix + pre/post projections
        residual_per_layer = (
            n_streams * n_streams  # mixing matrix
            + 2 * n_streams * d    # pre and post projection weights
        )

    per_layer_total = attn_per_layer + ffn_per_layer + norm_per_layer + residual_per_layer

    # Final norm
    if config.norm_type == "layernorm":
        final_norm_params = 2 * d
    elif config.norm_type == "dyt":
        final_norm_params = 2 * d + 1
    else:
        final_norm_params = d

    # Prediction head (weight-tied with embedding, so no extra params)
    head_params = 0  # Weight-tied

    # MTP auxiliary heads
    mtp_params = 0
    if config.prediction_type == "mtp":
        # Each aux head: linear(d, d) + norm(d) + linear(d, vocab)
        per_aux_head = d * d + d + d * vocab_size
        mtp_params = (config.mtp_n_heads - 1) * per_aux_head

    # mHC expand/collapse
    mhc_global_params = 0
    if config.residual_type == "mhc":
        n_streams = config.mhc_n_streams
        mhc_global_params = d * n_streams + n_streams * d  # expand + collapse

    total = (
        embedding_params
        + n_layers * per_layer_total
        + final_norm_params
        + head_params
        + mtp_params
        + mhc_global_params
    )

    return {
        "embedding": embedding_params,
        "attention_per_layer": attn_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "norm_per_layer": norm_per_layer,
        "residual_per_layer": residual_per_layer,
        "per_layer_total": per_layer_total,
        "final_norm": final_norm_params,
        "mtp_aux_heads": mtp_params,
        "mhc_global": mhc_global_params,
        "total": total,
    }


def estimate_memory(
    config: ModelConfig,
    training_config: TrainingConfig | None = None,
    batch_size: int | None = None,
    dtype: str = "bf16",
    optimizer: str = "adamw",
    activation_checkpointing: bool = True,
) -> dict[str, Any]:
    """Estimate total VRAM requirements for training.

    Parameters
    ----------
    config:
        Model architecture configuration.
    training_config:
        Training configuration (used for batch size if not overridden).
    batch_size:
        Override batch size.
    dtype:
        Parameter dtype (``"bf16"``, ``"fp16"``, ``"fp32"``).
    optimizer:
        Optimizer type (``"adamw"`` requires 2x param memory for moments).
    activation_checkpointing:
        Whether activation checkpointing is enabled.

    Returns
    -------
    dict[str, Any]
        Detailed memory breakdown in bytes and GB.
    """
    if training_config is None:
        training_config = TrainingConfig()

    if batch_size is None:
        batch_size = training_config.batch_size

    params = estimate_model_params(config)
    total_params = params["total"]
    bytes_per_param = BYTES_PER_PARAM.get(dtype, 2)

    # Model parameters memory
    param_bytes = total_params * bytes_per_param

    # Gradient memory (same dtype as parameters)
    gradient_bytes = total_params * bytes_per_param

    # Optimizer states
    if optimizer == "adamw":
        # AdamW: first moment (fp32) + second moment (fp32) = 8 bytes per param
        optimizer_bytes = total_params * 8
    elif optimizer == "sgd":
        optimizer_bytes = 0
    else:
        optimizer_bytes = total_params * 4

    # Activation memory estimation
    seq_len = config.max_seq_len
    d = config.d_model
    n_layers = config.n_layers

    # Per-layer activation memory (rough estimate)
    # Main activations: input, attention scores, FFN intermediates
    per_layer_activation = (
        batch_size * seq_len * d * 2  # Input + residual (bf16)
        + batch_size * config.n_heads * seq_len * seq_len * 2  # Attention scores
        + batch_size * seq_len * config.d_ff * 2  # FFN intermediate
    )

    # mHC multiplier: n_streams copies of the hidden state
    if config.residual_type == "mhc":
        per_layer_activation *= config.mhc_n_streams

    if activation_checkpointing:
        # With checkpointing, only store 1 activation per layer (recompute rest)
        activation_bytes = n_layers * batch_size * seq_len * d * bytes_per_param
    else:
        activation_bytes = n_layers * per_layer_activation

    # KV cache (not used during training, but relevant info)
    n_kv_heads = config.n_kv_heads or config.n_heads
    kv_cache_per_layer = 2 * batch_size * n_kv_heads * seq_len * config.head_dim * bytes_per_param
    kv_cache_total = n_layers * kv_cache_per_layer

    # Total
    total_bytes = param_bytes + gradient_bytes + optimizer_bytes + activation_bytes
    total_gb = total_bytes / (1024 ** 3)

    # Recommendations
    fits_24gb = total_gb < 22  # Leave 2GB headroom
    fits_48gb = total_gb < 44  # Leave 4GB headroom

    # Recommended batch sizes
    recommended_bs_24 = _recommend_batch_size(config, 24, dtype, optimizer, activation_checkpointing)
    recommended_bs_48 = _recommend_batch_size(config, 48, dtype, optimizer, activation_checkpointing)

    return {
        "total_params": total_params,
        "total_params_human": _human_readable(total_params),
        "param_breakdown": params,
        "param_bytes": param_bytes,
        "param_gb": param_bytes / (1024 ** 3),
        "gradient_bytes": gradient_bytes,
        "gradient_gb": gradient_bytes / (1024 ** 3),
        "optimizer_bytes": optimizer_bytes,
        "optimizer_gb": optimizer_bytes / (1024 ** 3),
        "activation_bytes": activation_bytes,
        "activation_gb": activation_bytes / (1024 ** 3),
        "kv_cache_bytes": kv_cache_total,
        "kv_cache_gb": kv_cache_total / (1024 ** 3),
        "total_bytes": total_bytes,
        "total_gb": total_gb,
        "fits_24gb": fits_24gb,
        "fits_48gb": fits_48gb,
        "recommended_batch_size_24gb": recommended_bs_24,
        "recommended_batch_size_48gb": recommended_bs_48,
        "activation_checkpointing": activation_checkpointing,
        "dtype": dtype,
        "batch_size": batch_size,
    }


def _recommend_batch_size(
    config: ModelConfig,
    gpu_gb: int,
    dtype: str,
    optimizer: str,
    activation_checkpointing: bool,
) -> int:
    """Binary search for the largest batch size that fits in GPU memory."""
    for bs in [64, 32, 16, 8, 4, 2, 1]:
        est = estimate_memory(
            config,
            batch_size=bs,
            dtype=dtype,
            optimizer=optimizer,
            activation_checkpointing=activation_checkpointing,
        )
        headroom = 4 if gpu_gb >= 48 else 2
        if est["total_gb"] < (gpu_gb - headroom):
            return bs
    return 1


def _human_readable(n: int) -> str:
    """Convert parameter count to human-readable format."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def print_report(result: dict[str, Any]) -> None:
    """Print a formatted memory estimation report."""
    print("\n" + "=" * 60)
    print("MiniGPT VRAM Estimation Report")
    print("=" * 60)
    print(f"\nTotal Parameters: {result['total_params_human']} ({result['total_params']:,})")
    print(f"Dtype: {result['dtype']}")
    print(f"Batch Size: {result['batch_size']}")
    print(f"Activation Checkpointing: {result['activation_checkpointing']}")
    print(f"\nMemory Breakdown:")
    print(f"  Parameters:      {result['param_gb']:.2f} GB")
    print(f"  Gradients:       {result['gradient_gb']:.2f} GB")
    print(f"  Optimizer:       {result['optimizer_gb']:.2f} GB")
    print(f"  Activations:     {result['activation_gb']:.2f} GB")
    print(f"  ---")
    print(f"  Total:           {result['total_gb']:.2f} GB")
    print(f"\nFits on 24GB GPU: {'Yes' if result['fits_24gb'] else 'NO'}")
    print(f"Fits on 48GB GPU: {'Yes' if result['fits_48gb'] else 'NO'}")
    print(f"\nRecommended batch size (24GB): {result['recommended_batch_size_24gb']}")
    print(f"Recommended batch size (48GB): {result['recommended_batch_size_48gb']}")
    print(f"\nKV Cache (inference): {result['kv_cache_gb']:.2f} GB")
    print("=" * 60 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="MiniGPT VRAM Estimator")
    parser.add_argument("--preset", type=str, default=None, help=f"Model preset: {list(PRESETS.keys())}")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Manual config overrides
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    args = parser.parse_args()

    if args.preset and args.preset in PRESETS:
        config = PRESETS[args.preset]
    else:
        config = ModelConfig()

    # Apply overrides
    if args.d_model:
        config.d_model = args.d_model
    if args.n_layers:
        config.n_layers = args.n_layers
    if args.n_heads:
        config.n_heads = args.n_heads
    if args.n_kv_heads:
        config.n_kv_heads = args.n_kv_heads

    result = estimate_memory(
        config,
        batch_size=args.batch_size,
        dtype=args.dtype,
        activation_checkpointing=not args.no_activation_checkpointing,
    )

    if args.json:
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
