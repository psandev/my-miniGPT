"""Convert ModelConfig presets to TorchTitan TOML training recipes.

Generates TOML files with the exact sections expected by TorchTitan:
``[model]``, ``[optimizer]``, ``[training]``, ``[metrics]``, ``[checkpoint]``.

Usage::

    python training/toml_generator.py --preset small --output /tmp/pretrain_small.toml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig, TrainingConfig, PRESETS

logger = logging.getLogger(__name__)


def model_config_to_toml_dict(
    model_config: ModelConfig,
    training_config: TrainingConfig | None = None,
    *,
    dataset_name: str = "smollm",
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    checkpoint_dir: str = "checkpoints",
    wandb_project: str = "minigpt",
) -> dict[str, dict]:
    """Convert model and training configs to a dict representing TOML sections.

    Parameters
    ----------
    model_config:
        Model architecture config.
    training_config:
        Training hyper-parameters. If ``None``, uses defaults.
    dataset_name:
        HuggingFace dataset identifier.
    tokenizer_name:
        HuggingFace tokenizer identifier.
    checkpoint_dir:
        Directory for saving checkpoints.
    wandb_project:
        WandB project name.

    Returns
    -------
    dict[str, dict]
        Nested dict with keys ``model``, ``optimizer``, ``training``,
        ``metrics``, ``checkpoint``.
    """
    if training_config is None:
        training_config = TrainingConfig()

    toml_dict = {
        "model": {
            "name": "minigpt",
            "preset": _find_preset_name(model_config),
            "vocab_size": model_config.vocab_size,
            "d_model": model_config.d_model,
            "n_layers": model_config.n_layers,
            "n_heads": model_config.n_heads,
            "n_kv_heads": model_config.n_kv_heads if model_config.n_kv_heads else model_config.n_heads,
            "max_seq_len": model_config.max_seq_len,
            "dropout": model_config.dropout,
            "attention_type": model_config.attention_type,
            "pos_encoding": model_config.pos_encoding,
            "norm_type": model_config.norm_type,
            "ffn_type": model_config.ffn_type,
            "residual_type": model_config.residual_type,
            "prediction_type": model_config.prediction_type,
            "use_bias": model_config.use_bias,
            "ffn_multiplier": model_config.ffn_multiplier,
            "norm_eps": model_config.norm_eps,
            "rope_base": model_config.rope_base,
            "tokenizer": tokenizer_name,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": training_config.lr,
            "beta1": training_config.beta1,
            "beta2": training_config.beta2,
            "weight_decay": training_config.weight_decay,
            "warmup_steps": training_config.warmup_steps,
            "scheduler": "cosine",
            "grad_clip": training_config.grad_clip,
        },
        "training": {
            "dataset": dataset_name,
            "batch_size": training_config.batch_size,
            "max_steps": training_config.max_steps,
            "seq_len": model_config.max_seq_len,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "mixed_precision": "bf16",
            "compile": training_config.compile,
            "activation_checkpointing": training_config.activation_checkpointing,
            "seed": 42,
        },
        "metrics": {
            "log_interval": training_config.log_interval,
            "wandb_project": wandb_project,
            "enable_wandb": True,
            "enable_tensorboard": True,
            "tensorboard_dir": f"{checkpoint_dir}/tb_logs",
        },
        "checkpoint": {
            "save_interval": training_config.save_interval,
            "checkpoint_dir": checkpoint_dir,
            "keep_last_n": 3,
        },
    }

    # Add MoE-specific fields if applicable
    if model_config.ffn_type == "moe":
        toml_dict["model"]["moe_num_experts"] = model_config.moe_num_experts
        toml_dict["model"]["moe_top_k"] = model_config.moe_top_k
        toml_dict["model"]["moe_aux_loss_weight"] = model_config.moe_aux_loss_weight

    # Add MLA-specific fields if applicable
    if model_config.attention_type == "mla":
        toml_dict["model"]["mla_kv_lora_rank"] = model_config.mla_kv_lora_rank
        toml_dict["model"]["mla_q_lora_rank"] = model_config.mla_q_lora_rank
        toml_dict["model"]["mla_rope_head_dim"] = model_config.mla_rope_head_dim

    # Add mHC-specific fields if applicable
    if model_config.residual_type == "mhc":
        toml_dict["model"]["mhc_n_streams"] = model_config.mhc_n_streams
        toml_dict["model"]["mhc_sinkhorn_iters"] = model_config.mhc_sinkhorn_iters

    # Add MTP-specific fields if applicable
    if model_config.prediction_type == "mtp":
        toml_dict["model"]["mtp_n_heads"] = model_config.mtp_n_heads
        toml_dict["model"]["mtp_loss_weight"] = model_config.mtp_loss_weight

    # Add DyT-specific fields if applicable
    if model_config.norm_type == "dyt":
        toml_dict["model"]["dyt_alpha_init"] = model_config.dyt_alpha_init

    # Add RoPE scaling if set
    if model_config.rope_scaling is not None:
        toml_dict["model"]["rope_scaling"] = model_config.rope_scaling

    return toml_dict


def _find_preset_name(config: ModelConfig) -> str:
    """Try to identify which preset this config matches."""
    for name, preset in PRESETS.items():
        if (config.d_model == preset.d_model
                and config.n_layers == preset.n_layers
                and config.n_heads == preset.n_heads):
            return name
    return "custom"


def dict_to_toml_string(toml_dict: dict[str, dict]) -> str:
    """Render a nested dict as a TOML string.

    Implements a minimal TOML serializer to avoid requiring a third-party
    library just for config generation.
    """
    lines = [
        "# MiniGPT TorchTitan Training Recipe",
        "# Auto-generated by toml_generator.py",
        "",
    ]

    for section, values in toml_dict.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, float):
                # Use scientific notation for very small numbers
                if abs(value) < 0.001 and value != 0.0:
                    lines.append(f"{key} = {value:.2e}")
                else:
                    lines.append(f"{key} = {value}")
            elif isinstance(value, int):
                lines.append(f"{key} = {value}")
            elif value is None:
                continue  # Skip None values
            else:
                lines.append(f"{key} = {value}")
        lines.append("")

    return "\n".join(lines)


def generate_toml(
    model_config: ModelConfig,
    training_config: TrainingConfig | None = None,
    output_path: str | None = None,
    **kwargs,
) -> str:
    """Generate a TorchTitan TOML recipe and optionally write to disk.

    Parameters
    ----------
    model_config:
        Model architecture configuration.
    training_config:
        Training hyper-parameters.
    output_path:
        If provided, write the TOML to this file.
    **kwargs:
        Forwarded to ``model_config_to_toml_dict`` (e.g. ``dataset_name``).

    Returns
    -------
    str
        The TOML content as a string.
    """
    toml_dict = model_config_to_toml_dict(model_config, training_config, **kwargs)
    toml_str = dict_to_toml_string(toml_dict)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(toml_str)
        logger.info("TOML recipe written to %s", output_path)

    return toml_str


def generate_all_preset_tomls(output_dir: str = "torchtitan_recipes") -> None:
    """Generate TOML recipes for all presets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    preset_to_file = {
        "tiny": "pretrain_tiny.toml",
        "small": "pretrain_small.toml",
        "medium": "pretrain_medium.toml",
    }

    for preset_name, filename in preset_to_file.items():
        if preset_name not in PRESETS:
            continue
        config = PRESETS[preset_name]
        generate_toml(config, output_path=str(out / filename))
        logger.info("Generated %s for preset '%s'", filename, preset_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate TorchTitan TOML recipes")
    parser.add_argument(
        "--preset", type=str, default=None,
        help=f"Preset name. Options: {list(PRESETS.keys())}",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--all", action="store_true", help="Generate TOMLs for all presets")
    parser.add_argument("--output-dir", type=str, default="torchtitan_recipes")
    args = parser.parse_args()

    if args.all:
        generate_all_preset_tomls(args.output_dir)
    elif args.preset:
        if args.preset not in PRESETS:
            logger.error("Unknown preset: %s", args.preset)
            sys.exit(1)
        config = PRESETS[args.preset]
        output = args.output or f"pretrain_{args.preset}.toml"
        toml_str = generate_toml(config, output_path=output)
        print(toml_str)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
