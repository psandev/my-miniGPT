"""Export MiniGPT checkpoint to HuggingFace Llama-compatible format.

Strips training-only components (MTP auxiliary heads, mHC streams) and
maps weights to a standard LlamaForCausalLM-compatible state dict for
use with HuggingFace transformers, vLLM, and other serving frameworks.

Usage::

    python quantization/export_hf.py --checkpoint checkpoints/final/checkpoint.pt \\
        --output-dir checkpoints/hf_export
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight mapping from MiniGPT -> Llama HF format
# ---------------------------------------------------------------------------

def _build_weight_map(config: ModelConfig, n_layers: int) -> dict[str, str]:
    """Build a mapping from MiniGPT state dict keys to Llama HF keys.

    Returns
    -------
    dict[str, str]
        Mapping of ``minigpt_key -> hf_key``.
    """
    mapping = {
        "tok_emb.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "final_norm.weight": "model.norm.weight",
    }

    # If there is a LayerNorm-style norm with a bias
    if config.norm_type == "layernorm":
        mapping["final_norm.bias"] = "model.norm.bias"

    for i in range(n_layers):
        prefix_src = f"layers.{i}"
        prefix_dst = f"model.layers.{i}"

        # Attention
        mapping.update({
            f"{prefix_src}.attn.q_proj.weight": f"{prefix_dst}.self_attn.q_proj.weight",
            f"{prefix_src}.attn.k_proj.weight": f"{prefix_dst}.self_attn.k_proj.weight",
            f"{prefix_src}.attn.v_proj.weight": f"{prefix_dst}.self_attn.v_proj.weight",
            f"{prefix_src}.attn.o_proj.weight": f"{prefix_dst}.self_attn.o_proj.weight",
        })

        # Attention norms
        mapping[f"{prefix_src}.attn_norm.weight"] = f"{prefix_dst}.input_layernorm.weight"
        mapping[f"{prefix_src}.ffn_norm.weight"] = f"{prefix_dst}.post_attention_layernorm.weight"

        if config.norm_type == "layernorm":
            mapping[f"{prefix_src}.attn_norm.bias"] = f"{prefix_dst}.input_layernorm.bias"
            mapping[f"{prefix_src}.ffn_norm.bias"] = f"{prefix_dst}.post_attention_layernorm.bias"

        # FFN - map to Llama's gate_proj/up_proj/down_proj naming
        if config.ffn_type == "swiglu":
            mapping.update({
                f"{prefix_src}.ffn.w_gate.weight": f"{prefix_dst}.mlp.gate_proj.weight",
                f"{prefix_src}.ffn.w_up.weight": f"{prefix_dst}.mlp.up_proj.weight",
                f"{prefix_src}.ffn.w_down.weight": f"{prefix_dst}.mlp.down_proj.weight",
            })
        elif config.ffn_type in ("gelu", "relu"):
            mapping.update({
                f"{prefix_src}.ffn.w_up.weight": f"{prefix_dst}.mlp.up_proj.weight",
                f"{prefix_src}.ffn.w_down.weight": f"{prefix_dst}.mlp.down_proj.weight",
            })
        elif config.ffn_type == "moe":
            # For MoE, export only the first expert (simplified)
            # In practice, you'd want a dedicated MoE export format
            mapping.update({
                f"{prefix_src}.ffn.router.weight": f"{prefix_dst}.mlp.gate.weight",
            })
            for e in range(config.moe_num_experts):
                mapping.update({
                    f"{prefix_src}.ffn.experts.{e}.w_gate.weight": f"{prefix_dst}.mlp.experts.{e}.gate_proj.weight",
                    f"{prefix_src}.ffn.experts.{e}.w_up.weight": f"{prefix_dst}.mlp.experts.{e}.up_proj.weight",
                    f"{prefix_src}.ffn.experts.{e}.w_down.weight": f"{prefix_dst}.mlp.experts.{e}.down_proj.weight",
                })

    return mapping


def _strip_mtp_heads(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove MTP auxiliary head weights, keeping only the main head.

    MTP heads have keys like ``mtp_heads.0.*``, ``mtp_heads.1.*``, etc.
    """
    filtered = {}
    stripped_count = 0
    for key, value in state_dict.items():
        if key.startswith("mtp_heads.") or key.startswith("mtp_"):
            stripped_count += 1
            continue
        filtered[key] = value

    if stripped_count > 0:
        logger.info("Stripped %d MTP auxiliary head parameters", stripped_count)
    return filtered


def _collapse_mhc_streams(
    state_dict: dict[str, torch.Tensor],
    config: ModelConfig,
) -> dict[str, torch.Tensor]:
    """Collapse mHC multi-stream residual parameters.

    The mHC expand/collapse projections and mixing matrices are removed.
    Stream expansion/collapse is not needed at inference time since the
    exported model uses standard residual connections.
    """
    filtered = {}
    stripped_count = 0

    mhc_prefixes = ("mhc_", "stream_expand", "stream_collapse", "residual.")

    for key, value in state_dict.items():
        is_mhc = any(key.startswith(p) or f".{p}" in key for p in mhc_prefixes)
        if is_mhc and "mixing" in key:
            stripped_count += 1
            continue
        if "residual.alpha" in key or "residual.expand" in key or "residual.collapse" in key:
            stripped_count += 1
            continue
        filtered[key] = value

    if stripped_count > 0:
        logger.info("Stripped %d mHC residual stream parameters", stripped_count)
    return filtered


def _create_hf_config(config: ModelConfig) -> dict[str, Any]:
    """Create a HuggingFace-compatible config.json for the exported model."""
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "intermediate_size": config.d_ff,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads or config.n_heads,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_base,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }

    if config.rope_scaling is not None:
        hf_config["rope_scaling"] = {
            "type": "linear",
            "factor": config.rope_scaling,
        }

    return hf_config


def export_to_hf(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    dtype: str = "bfloat16",
) -> Path:
    """Export a MiniGPT checkpoint to HuggingFace format.

    Parameters
    ----------
    checkpoint_path:
        Path to the MiniGPT training checkpoint (``.pt`` file).
    output_dir:
        Directory to write the HF-format model.
    tokenizer_name:
        Tokenizer to copy into the output directory.
    dtype:
        Target dtype for saved weights.

    Returns
    -------
    Path
        Path to the output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**ckpt["model_config"])
    state_dict = ckpt["model"]

    # Strip training-only components
    state_dict = _strip_mtp_heads(state_dict)
    if model_config.residual_type == "mhc":
        state_dict = _collapse_mhc_streams(state_dict, model_config)

    # Build weight mapping and rename keys
    weight_map = _build_weight_map(model_config, model_config.n_layers)
    hf_state_dict: dict[str, torch.Tensor] = {}
    unmapped_keys: list[str] = []

    for src_key, tensor in state_dict.items():
        if src_key in weight_map:
            dst_key = weight_map[src_key]
            hf_state_dict[dst_key] = tensor
        else:
            # Keep unmapped keys with their original names as a fallback
            unmapped_keys.append(src_key)
            hf_state_dict[src_key] = tensor

    if unmapped_keys:
        logger.warning("Unmapped keys (kept as-is): %s", unmapped_keys[:20])

    # Convert dtype
    target_dtype = getattr(torch, dtype, torch.bfloat16)
    for key in hf_state_dict:
        if hf_state_dict[key].is_floating_point():
            hf_state_dict[key] = hf_state_dict[key].to(target_dtype)

    # Save model weights
    from safetensors.torch import save_file

    save_file(hf_state_dict, str(out_path / "model.safetensors"))
    logger.info("Saved model weights to %s", out_path / "model.safetensors")

    # Save config
    hf_config = _create_hf_config(model_config)
    with open(out_path / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Save generation config
    gen_config = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
    }
    with open(out_path / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    # Copy tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.save_pretrained(str(out_path))
        logger.info("Tokenizer saved to %s", out_path)
    except Exception as exc:
        logger.warning("Could not save tokenizer: %s", exc)

    # Save original MiniGPT config for reference
    with open(out_path / "minigpt_config.json", "w") as f:
        json.dump(model_config.__dict__, f, indent=2, default=str)

    logger.info("HuggingFace export complete: %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export MiniGPT to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="MiniGPT checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF model")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    export_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
