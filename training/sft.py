"""Supervised Fine-Tuning (SFT) for MiniGPT.

Instruction-following fine-tuning with:
- Lower learning rate (1e-5 default)
- Chat template formatting
- Instruction/response masking (loss only on response tokens)

Usage::

    python training/sft.py --checkpoint checkpoints/final/checkpoint.pt \\
        --dataset alpaca --output-dir checkpoints/sft
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig, TrainingConfig
from miniGPT.model import MiniGPT
from training.data import build_sft_dataloader, load_tokenizer
from training.torchtitan_integration import get_autocast_context

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


def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay with linear warmup for SFT."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def run_sft(
    checkpoint_path: str,
    dataset_name: str = "alpaca",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    output_dir: str = "checkpoints/sft",
    lr: float = 1e-5,
    batch_size: int = 4,
    max_steps: int = 5000,
    warmup_steps: int = 200,
    max_seq_len: int = 2048,
    grad_clip: float = 1.0,
    gradient_accumulation_steps: int = 4,
    log_interval: int = 10,
    save_interval: int = 500,
    use_wandb: bool = True,
) -> Path:
    """Run supervised fine-tuning on a pretrained checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to the pretrained model checkpoint.
    dataset_name:
        SFT dataset (``"alpaca"``, ``"oasst"``, or HF path).
    tokenizer_name:
        Tokenizer identifier.
    output_dir:
        Directory for SFT checkpoints.
    lr:
        Learning rate (lower than pretraining; default 1e-5).
    batch_size:
        Micro-batch size.
    max_steps:
        Total SFT training steps.
    warmup_steps:
        Linear warmup steps.
    max_seq_len:
        Maximum sequence length.
    grad_clip:
        Maximum gradient norm.
    gradient_accumulation_steps:
        Number of gradient accumulation steps.
    log_interval:
        Steps between log messages.
    save_interval:
        Steps between checkpoints.
    use_wandb:
        Whether to log to WandB.

    Returns
    -------
    Path
        Path to the final SFT checkpoint directory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load pretrained checkpoint
    logger.info("Loading pretrained checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**ckpt["model_config"])
    model_config.max_seq_len = max_seq_len

    model = MiniGPT(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    logger.info("Model loaded with %s parameters", model.count_parameters().get("total_human", "?"))

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule(optimizer, warmup_steps, max_steps)

    # Data
    dataloader = build_sft_dataloader(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    # WandB
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project="minigpt-sft",
                config={
                    "lr": lr,
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "dataset": dataset_name,
                    "model_config": model_config.__dict__,
                },
            )
        except ImportError:
            logger.warning("wandb not installed")
            use_wandb = False

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    step = 0
    running_loss = 0.0
    t_start = time.time()

    logger.info("Starting SFT training for %d steps", max_steps)

    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with get_autocast_context(device_type=device):
                outputs = model(input_ids, targets=labels)

            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        step += 1
        running_loss += accum_loss

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - t_start
            logger.info("step=%d  loss=%.4f  lr=%.2e  time=%.1fs", step, avg_loss, scheduler.get_last_lr()[0], elapsed)
            if use_wandb:
                import wandb

                wandb.log({"sft/loss": avg_loss, "sft/lr": scheduler.get_last_lr()[0], "sft/step": step})
            running_loss = 0.0
            t_start = time.time()

        if step % save_interval == 0:
            ckpt_dir = out_path / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "model_config": model_config.__dict__,
                },
                ckpt_dir / "checkpoint.pt",
            )

    # Final save
    final_dir = out_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "step": step, "model_config": model_config.__dict__},
        final_dir / "checkpoint.pt",
    )
    with open(final_dir / "config.json", "w") as f:
        json.dump(model_config.__dict__, f, indent=2, default=str)

    if use_wandb:
        import wandb

        wandb.finish()

    logger.info("SFT complete. Final checkpoint: %s", final_dir)
    return final_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load central config — all SFT parameters default from config.yaml
    cfg = _load_cfg()
    _s = cfg.sft    if cfg else None
    _p = cfg.paths  if cfg else None

    parser = argparse.ArgumentParser(description="MiniGPT Supervised Fine-Tuning")
    parser.add_argument("--checkpoint", default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--dataset",    default=None, help="SFT dataset name")
    parser.add_argument("--tokenizer",  default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--max-steps",  type=int,   default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-seq-len",  type=int, default=None)
    parser.add_argument("--grad-clip",    type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # Resolve: CLI arg → config.yaml → hard default
    checkpoint = args.checkpoint or (str(_p.pretrain_checkpoint) if _p else None)
    if not checkpoint:
        logger.error("No checkpoint provided. Set paths.pretrain_checkpoint in config.yaml or pass --checkpoint.")
        sys.exit(1)

    run_sft(
        checkpoint_path=checkpoint,
        dataset_name  = args.dataset    or (str(_s.dataset)   if _s else "alpaca"),
        tokenizer_name= args.tokenizer  or (str(cfg.data.tokenizer) if cfg else DEFAULT_TOKENIZER),
        output_dir    = args.output_dir or (str(Path(_p.checkpoints_dir) / "sft") if _p else "checkpoints/sft"),
        lr            = args.lr            or (float(_s.lr)           if _s else 1e-5),
        batch_size    = args.batch_size    or (int(_s.batch_size)     if _s else 4),
        max_steps     = args.max_steps     or (int(_s.max_steps)      if _s else 5000),
        warmup_steps  = args.warmup_steps  or (int(_s.warmup_steps)   if _s else 200),
        max_seq_len   = args.max_seq_len   or (int(_s.max_seq_len)    if _s else 2048),
        grad_clip     = args.grad_clip     or (float(_s.grad_clip)    if _s else 1.0),
        gradient_accumulation_steps = args.gradient_accumulation_steps or (int(_s.gradient_accumulation_steps) if _s else 4),
        log_interval  = int(_s.log_interval)  if _s else 10,
        save_interval = int(_s.save_interval) if _s else 500,
        use_wandb     = not args.no_wandb and (bool(_s.use_wandb) if _s else False),
    )


if __name__ == "__main__":
    main()
