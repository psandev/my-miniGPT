"""Download and prepare training datasets for MiniGPT.

Downloads SmolLM-Corpus, TinyStories, C4, OpenAssistant, Alpaca,
and Anthropic HH-RLHF via the HuggingFace datasets library.

Usage::

    python scripts/download_data.py --datasets tinystories,alpaca --output-dir data/
    python scripts/download_data.py --all --output-dir data/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

AVAILABLE_DATASETS = {
    "smollm": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "fineweb-edu-dedup",
        "description": "SmolLM-Corpus FineWeb-Edu-dedup (pretraining)",
    },
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "description": "TinyStories (small-scale pretraining/testing)",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "description": "C4 English (general pretraining)",
    },
    "oasst": {
        "path": "OpenAssistant/oasst1",
        "description": "OpenAssistant OASST1 (SFT)",
    },
    "oasst2": {
        "path": "OpenAssistant/oasst2",
        "description": "OpenAssistant OASST2 (SFT)",
    },
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "description": "Stanford Alpaca (SFT)",
    },
    "hh_rlhf": {
        "path": "Anthropic/hh-rlhf",
        "description": "Anthropic HH-RLHF (DPO alignment)",
    },
}


def download_dataset(
    name: str,
    output_dir: str = "data",
    streaming_preview: bool = False,
    max_preview_samples: int = 5,
) -> None:
    """Download a single dataset.

    Parameters
    ----------
    name:
        Dataset name from AVAILABLE_DATASETS.
    output_dir:
        Root directory for downloaded data.
    streaming_preview:
        If True, only stream and preview a few samples (don't save).
    max_preview_samples:
        Number of samples to preview in streaming mode.
    """
    from datasets import load_dataset

    if name not in AVAILABLE_DATASETS:
        logger.error("Unknown dataset: %s. Available: %s", name, list(AVAILABLE_DATASETS.keys()))
        return

    info = AVAILABLE_DATASETS[name]
    logger.info("Downloading %s: %s", name, info["description"])

    kwargs = {
        "path": info["path"],
        "split": "train",
        "trust_remote_code": True,
    }
    if "name" in info:
        kwargs["name"] = info["name"]

    if streaming_preview:
        kwargs["streaming"] = True
        ds = load_dataset(**kwargs)
        logger.info("Preview of %s:", name)
        for i, example in enumerate(ds):
            if i >= max_preview_samples:
                break
            # Print first 200 chars of each field
            for key, value in example.items():
                text = str(value)[:200]
                logger.info("  [%d] %s: %s", i, key, text)
        return

    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(**kwargs)
    ds.save_to_disk(str(out_path))
    logger.info("Dataset '%s' saved to %s (%d examples)", name, out_path, len(ds))


def download_all(output_dir: str = "data", skip_large: bool = True) -> None:
    """Download all available datasets.

    Parameters
    ----------
    output_dir:
        Root directory for downloaded data.
    skip_large:
        If True, skip very large datasets (C4, SmolLM-Corpus) and only
        download manageable ones.
    """
    large_datasets = {"c4", "smollm"}

    for name in AVAILABLE_DATASETS:
        if skip_large and name in large_datasets:
            logger.info("Skipping large dataset '%s' (use --no-skip-large to include)", name)
            continue
        try:
            download_dataset(name, output_dir)
        except Exception as exc:
            logger.error("Failed to download '%s': %s", name, exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download MiniGPT training datasets")
    parser.add_argument(
        "--datasets", type=str, default=None,
        help=f"Comma-separated dataset names. Available: {list(AVAILABLE_DATASETS.keys())}",
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--preview", action="store_true", help="Stream preview only")
    parser.add_argument("--no-skip-large", action="store_true", help="Include large datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, info in AVAILABLE_DATASETS.items():
            print(f"  {name:15s} - {info['description']}")
        return

    if args.all:
        download_all(args.output_dir, skip_large=not args.no_skip_large)
    elif args.datasets:
        for name in args.datasets.split(","):
            name = name.strip()
            download_dataset(name, args.output_dir, streaming_preview=args.preview)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
