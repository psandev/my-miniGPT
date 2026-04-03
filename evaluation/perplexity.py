"""WikiText-2 and C4 validation perplexity computation for MiniGPT.

Computes perplexity on standard validation sets using a sliding window
approach for sequences longer than the model's context length.

Usage::

    python evaluation/perplexity.py --checkpoint checkpoints/final/checkpoint.pt \\
        --dataset wikitext2 --output results/perplexity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig
from miniGPT.model import MiniGPT

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: MiniGPT,
    token_ids: torch.Tensor,
    max_seq_len: int,
    stride: int | None = None,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute perplexity over a sequence of token IDs.

    Uses a sliding window approach to handle sequences longer than
    the model's maximum context length.

    Parameters
    ----------
    model:
        MiniGPT model in eval mode.
    token_ids:
        1-D tensor of token IDs.
    max_seq_len:
        Maximum sequence length for the model.
    stride:
        Sliding window stride. Defaults to ``max_seq_len // 2``.
    device:
        Device for computation.

    Returns
    -------
    dict[str, float]
        Dictionary with ``perplexity``, ``loss``, ``num_tokens``.
    """
    if stride is None:
        stride = max_seq_len // 2

    model.eval()
    token_ids = token_ids.to(device)
    seq_len = token_ids.size(0)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for begin in range(0, seq_len - 1, stride):
            end = min(begin + max_seq_len, seq_len)
            input_ids = token_ids[begin:end - 1].unsqueeze(0)
            targets = token_ids[begin + 1:end].unsqueeze(0)

            # Only count loss for tokens in the second half of the window
            # (to avoid re-counting overlapping regions)
            target_len = end - begin - 1
            if begin > 0:
                # Overlap region: only count the new tokens
                overlap = max_seq_len - stride
                count_from = min(overlap, target_len)
            else:
                count_from = 0

            outputs = model(input_ids, targets=targets)
            logits = outputs["logits"]

            # Compute per-token loss for the counted region
            shift_logits = logits[:, count_from:, :].contiguous()
            shift_targets = targets[:, count_from:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                reduction="sum",
            )

            n_tokens = shift_targets.numel()
            total_loss += loss.item()
            total_tokens += n_tokens

            if end >= seq_len:
                break

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "num_tokens": total_tokens,
    }


def evaluate_perplexity(
    checkpoint_path: str,
    dataset_name: str = "wikitext2",
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    max_samples: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate perplexity of a MiniGPT checkpoint on a validation set.

    Parameters
    ----------
    checkpoint_path:
        Path to MiniGPT checkpoint.
    dataset_name:
        Dataset to evaluate on: ``"wikitext2"`` or ``"c4"``.
    tokenizer_name:
        Tokenizer for encoding text.
    max_samples:
        Maximum number of documents to process (for speed).
    device:
        Device to use. Defaults to CUDA if available.

    Returns
    -------
    dict[str, Any]
        Perplexity results including the dataset name and model info.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**ckpt["model_config"])
    model = MiniGPT(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load dataset
    token_ids = _load_validation_tokens(dataset_name, tokenizer, max_samples)
    logger.info("Evaluation dataset: %s (%d tokens)", dataset_name, len(token_ids))

    # Compute perplexity
    result = compute_perplexity(
        model, token_ids,
        max_seq_len=model_config.max_seq_len,
        device=device,
    )

    result["dataset"] = dataset_name
    result["checkpoint"] = checkpoint_path
    result["model_params"] = model.count_parameters().get("total", 0)

    logger.info(
        "Perplexity on %s: %.2f (loss: %.4f, tokens: %d)",
        dataset_name, result["perplexity"], result["loss"], result["num_tokens"],
    )

    return result


def _load_validation_tokens(
    dataset_name: str,
    tokenizer: Any,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Load and tokenize a validation dataset."""
    from datasets import load_dataset

    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text_field = "text"
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        text_field = "text"
    else:
        ds = load_dataset(dataset_name, split="validation")
        text_field = "text"

    all_tokens: list[int] = []
    count = 0
    for example in ds:
        text = example.get(text_field, "")
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        count += 1
        if max_samples and count >= max_samples:
            break

    return torch.tensor(all_tokens, dtype=torch.long)


def log_to_wandb(result: dict[str, Any]) -> None:
    """Log perplexity result to WandB."""
    try:
        import wandb

        if wandb.run is None:
            wandb.init(project="minigpt-eval")

        wandb.log({
            f"perplexity/{result['dataset']}": result["perplexity"],
            f"loss/{result['dataset']}": result["loss"],
        })
    except ImportError:
        logger.warning("wandb not installed")


def _load_cfg():
    """Load central config.yaml via OmegaConf (silent fallback if missing)."""
    try:
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        return OmegaConf.load(cfg_path)
    except Exception:
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load central config — perplexity parameters default from config.yaml
    cfg = _load_cfg()
    _e = cfg.eval   if cfg else None
    _p = cfg.paths  if cfg else None

    parser = argparse.ArgumentParser(description="MiniGPT Perplexity Evaluation")
    parser.add_argument("--checkpoint",   default=None, help="Path to MiniGPT checkpoint")
    parser.add_argument("--dataset",      default=None, choices=["wikitext2", "c4"])
    parser.add_argument("--tokenizer",    default=None)
    parser.add_argument("--max-samples",  type=int, default=None)
    parser.add_argument("--output",       type=str, default=None)
    parser.add_argument("--wandb",        action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (str(_p.pretrain_checkpoint) if _p else None)
    if not checkpoint:
        logger.error("No checkpoint provided. Set paths.pretrain_checkpoint in config.yaml or pass --checkpoint.")
        sys.exit(1)

    output_path = args.output or (str(Path(_p.results_dir) / "perplexity.json") if _p else None)

    result = evaluate_perplexity(
        checkpoint_path=checkpoint,
        dataset_name   = args.dataset     or (str(_e.perplexity_dataset) if _e else "wikitext2"),
        tokenizer_name = args.tokenizer   or (str(cfg.data.tokenizer) if cfg else "HuggingFaceTB/cosmo2-tokenizer"),
        max_samples    = args.max_samples or (int(_e.perplexity_max_samples) if _e and _e.perplexity_max_samples else None),
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    use_wandb = args.wandb or (bool(_e.use_wandb) if _e else False)
    if use_wandb:
        log_to_wandb(result)


if __name__ == "__main__":
    main()
