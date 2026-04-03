"""Direct Preference Optimization (DPO) alignment for MiniGPT.

Uses the TRL library's DPOTrainer to align the model on preference pairs
(chosen/rejected) from Anthropic HH-RLHF or compatible datasets.

Usage::

    python training/dpo.py --checkpoint checkpoints/sft/final/checkpoint.pt \\
        --output-dir checkpoints/dpo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig
from miniGPT.model import MiniGPT

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "HuggingFaceTB/cosmo2-tokenizer"


def _load_cfg():
    """Load central config.yaml via OmegaConf (silent fallback if missing)."""
    try:
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        return OmegaConf.load(cfg_path)
    except Exception:
        return None


def _load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> tuple[MiniGPT, ModelConfig]:
    """Load a MiniGPT model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**ckpt["model_config"])
    model = MiniGPT(model_config)
    model.load_state_dict(ckpt["model"])
    return model, model_config


def prepare_dpo_dataset(
    dataset_name: str = "hh_rlhf",
    max_samples: int | None = None,
) -> list[dict[str, str]]:
    """Load and format a preference dataset for DPO training.

    Returns a list of dicts with keys: ``prompt``, ``chosen``, ``rejected``.
    """
    from datasets import load_dataset

    dataset_map = {
        "hh_rlhf": ("Anthropic/hh-rlhf", None),
    }

    path, name = dataset_map.get(dataset_name, (dataset_name, None))
    ds = load_dataset(path, name=name, split="train")

    records = []
    for example in ds:
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        if not chosen or not rejected:
            continue

        # Extract prompt and responses from HH-RLHF format
        prompt = ""
        chosen_response = chosen
        rejected_response = rejected

        if "\n\nAssistant:" in chosen:
            parts = chosen.rsplit("\n\nAssistant:", 1)
            prompt = parts[0]
            chosen_response = parts[1].strip() if len(parts) > 1 else chosen

        if "\n\nAssistant:" in rejected:
            parts = rejected.rsplit("\n\nAssistant:", 1)
            rejected_response = parts[1].strip() if len(parts) > 1 else rejected

        records.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })

        if max_samples and len(records) >= max_samples:
            break

    logger.info("Loaded %d preference pairs from %s", len(records), dataset_name)
    return records


def run_dpo(
    checkpoint_path: str,
    dataset_name: str = "hh_rlhf",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    output_dir: str = "checkpoints/dpo",
    beta: float = 0.1,
    lr: float = 5e-6,
    batch_size: int = 4,
    max_steps: int = 2000,
    max_seq_len: int = 1024,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    max_samples: int | None = None,
    use_wandb: bool = True,
) -> Path:
    """Run DPO alignment training.

    Parameters
    ----------
    checkpoint_path:
        Path to the SFT checkpoint to align.
    dataset_name:
        Preference dataset name (``"hh_rlhf"`` or HF path).
    tokenizer_name:
        Tokenizer identifier.
    output_dir:
        Directory for DPO checkpoints.
    beta:
        KL divergence penalty coefficient.
    lr:
        Learning rate for DPO (typically very low).
    batch_size:
        Per-device batch size.
    max_steps:
        Maximum training steps.
    max_seq_len:
        Maximum sequence length for tokenization.
    gradient_accumulation_steps:
        Gradient accumulation steps.
    warmup_steps:
        Warmup steps for the scheduler.
    max_samples:
        Optional cap on dataset size.
    use_wandb:
        Whether to log to WandB.

    Returns
    -------
    Path
        Path to the final DPO checkpoint.
    """
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the SFT model
    logger.info("Loading SFT checkpoint from %s", checkpoint_path)
    model, model_config = _load_model_from_checkpoint(checkpoint_path, device="cpu")

    # Load a reference copy for DPO
    ref_model, _ = _load_model_from_checkpoint(checkpoint_path, device="cpu")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    records = prepare_dpo_dataset(dataset_name, max_samples=max_samples)
    dataset = Dataset.from_list(records)

    # DPO training configuration
    dpo_config = DPOConfig(
        output_dir=str(out_path),
        beta=beta,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        max_length=max_seq_len,
        max_prompt_length=max_seq_len // 2,
        logging_steps=10,
        save_steps=500,
        bf16=torch.cuda.is_available(),
        report_to="wandb" if use_wandb else "none",
        run_name="minigpt-dpo",
        remove_unused_columns=False,
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training for %d steps (beta=%.3f, lr=%.2e)", max_steps, beta, lr)
    trainer.train()

    # Save final
    final_dir = out_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model_config.__dict__,
        },
        final_dir / "checkpoint.pt",
    )
    with open(final_dir / "config.json", "w") as f:
        json.dump(model_config.__dict__, f, indent=2, default=str)

    logger.info("DPO training complete. Final checkpoint: %s", final_dir)
    return final_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load central config — all DPO parameters default from config.yaml
    cfg = _load_cfg()
    _d = cfg.dpo    if cfg else None
    _p = cfg.paths  if cfg else None

    parser = argparse.ArgumentParser(description="MiniGPT DPO Alignment")
    parser.add_argument("--checkpoint",   default=None, help="Path to SFT checkpoint")
    parser.add_argument("--dataset",      default=None, help="Preference dataset name")
    parser.add_argument("--tokenizer",    default=None)
    parser.add_argument("--output-dir",   default=None)
    parser.add_argument("--beta",         type=float, default=None, help="KL divergence penalty")
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--max-steps",    type=int,   default=None)
    parser.add_argument("--max-seq-len",  type=int,   default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int,   default=None)
    parser.add_argument("--max-samples",  type=int,   default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (str(_p.sft_checkpoint) if _p else None)
    if not checkpoint:
        logger.error("No checkpoint provided. Set paths.sft_checkpoint in config.yaml or pass --checkpoint.")
        sys.exit(1)

    run_dpo(
        checkpoint_path=checkpoint,
        dataset_name  = args.dataset   or (str(_d.dataset)    if _d else "hh_rlhf"),
        tokenizer_name= args.tokenizer or (str(cfg.data.tokenizer) if cfg else DEFAULT_TOKENIZER),
        output_dir    = args.output_dir or (str(Path(_p.checkpoints_dir) / "dpo") if _p else "checkpoints/dpo"),
        beta          = args.beta          or (float(_d.beta)         if _d else 0.1),
        lr            = args.lr            or (float(_d.lr)           if _d else 5e-6),
        batch_size    = args.batch_size    or (int(_d.batch_size)     if _d else 4),
        max_steps     = args.max_steps     or (int(_d.max_steps)      if _d else 2000),
        max_seq_len   = args.max_seq_len   or (int(_d.max_seq_len)    if _d else 1024),
        gradient_accumulation_steps = args.gradient_accumulation_steps or (int(_d.gradient_accumulation_steps) if _d else 4),
        warmup_steps  = args.warmup_steps  or (int(_d.warmup_steps)   if _d else 100),
        max_samples   = args.max_samples   or (int(_d.max_samples) if _d and _d.max_samples else None),
        use_wandb     = not args.no_wandb and (bool(_d.use_wandb) if _d else False),
    )


if __name__ == "__main__":
    main()
