"""Optional: Train a custom byte-level BPE tokenizer using the tokenizers library.

By default, MiniGPT reuses an existing tokenizer (Llama 3.2 1B). This script
provides the option to train a domain-specific tokenizer from scratch.

Usage::

    python scripts/train_tokenizer.py --dataset tinystories --vocab-size 32000 \\
        --output tokenizer/custom_bpe
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def data_iterator(
    dataset_name: str,
    text_field: str = "text",
    max_samples: int | None = None,
) -> Iterator[str]:
    """Iterate over text from a HuggingFace dataset.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset path or name from the project registry.
    text_field:
        Name of the text field in the dataset.
    max_samples:
        Maximum number of samples to process.

    Yields
    ------
    str
        Text documents from the dataset.
    """
    from datasets import load_dataset

    dataset_map = {
        "tinystories": ("roneneldan/TinyStories", None),
        "smollm": ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup"),
        "c4": ("allenai/c4", "en"),
    }

    path, name = dataset_map.get(dataset_name, (dataset_name, None))

    ds = load_dataset(path, name=name, split="train", streaming=True, trust_remote_code=True)

    count = 0
    for example in ds:
        text = example.get(text_field, "")
        if text and text.strip():
            yield text
            count += 1
            if max_samples and count >= max_samples:
                return


def train_tokenizer(
    dataset_name: str = "tinystories",
    vocab_size: int = 32000,
    min_frequency: int = 2,
    output_dir: str = "tokenizer/custom_bpe",
    max_samples: int = 100000,
    special_tokens: list[str] | None = None,
) -> Path:
    """Train a byte-level BPE tokenizer.

    Parameters
    ----------
    dataset_name:
        Dataset to train on.
    vocab_size:
        Target vocabulary size.
    min_frequency:
        Minimum token frequency for inclusion.
    output_dir:
        Directory to save the trained tokenizer.
    max_samples:
        Maximum training samples.
    special_tokens:
        Special tokens to include.

    Returns
    -------
    Path
        Path to the saved tokenizer directory.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer
    from tokenizers.processors import ByteLevel as ByteLevelPostProcessor

    if special_tokens is None:
        special_tokens = [
            "<|endoftext|>",
            "<|padding|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
        ]

    logger.info(
        "Training BPE tokenizer: vocab_size=%d, dataset=%s, max_samples=%d",
        vocab_size, dataset_name, max_samples,
    )

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelPostProcessor()

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train
    logger.info("Starting tokenizer training...")
    tokenizer.train_from_iterator(
        data_iterator(dataset_name, max_samples=max_samples),
        trainer=trainer,
    )

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer_path = out_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info("Tokenizer saved to %s (vocab_size=%d)", tokenizer_path, tokenizer.get_vocab_size())

    # Also save in HuggingFace format for compatibility
    try:
        from transformers import PreTrainedTokenizerFast

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|padding|>",
        )
        hf_tokenizer.save_pretrained(str(out_path))
        logger.info("HuggingFace-format tokenizer saved to %s", out_path)
    except ImportError:
        logger.warning("transformers not installed; skipping HF-format save")

    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer for MiniGPT")
    parser.add_argument("--dataset", default="tinystories", help="Training dataset name")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", default="tokenizer/custom_bpe")
    parser.add_argument("--max-samples", type=int, default=100000)
    args = parser.parse_args()

    train_tokenizer(
        dataset_name=args.dataset,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
