"""Download and prepare all MiniGPT training datasets.

This script handles the full data pipeline for every training stage:
  - Pretraining  : SmolLM-Corpus, TinyStories, C4
  - SFT          : Alpaca, OpenAssistant (OASST1/2)
  - DPO          : Anthropic HH-RLHF

For PRETRAINING datasets the pipeline is:
  download → tokenize → pack into fixed-length chunks → save as binary .bin
  (uint16 token IDs, same layout as nanoGPT — memory-mappable at training time)

For SFT/DPO datasets the pipeline is:
  download → format with chat template → tokenize → mask instruction tokens
  → save as .jsonl  (one JSON object per line, human-readable and streamable)

Output layout
-------------
/home/access/peter/trd/miniGPT/data/
├── tinystories/
│   ├── train.bin              <- packed uint16 token IDs
│   ├── val.bin
│   └── meta.json              <- n_train_tokens, n_val_tokens, vocab_size, ...
├── smollm/
│   ├── train.bin
│   ├── val.bin
│   └── meta.json
├── c4/
│   ├── train.bin
│   ├── val.bin
│   └── meta.json
├── alpaca/
│   ├── train.jsonl            <- {input_ids, labels, attention_mask}
│   ├── val.jsonl
│   └── meta.json              <- n_train, n_val, avg_len, ...
├── oasst/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── meta.json
├── oasst2/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── meta.json
└── hh_rlhf/
    ├── train.jsonl            <- {prompt, chosen, rejected}
    ├── val.jsonl
    └── meta.json

Usage
-----
# Prepare a single small dataset (fast, good for testing)
python data/prepare.py --datasets tinystories

# Prepare SFT + DPO datasets
python data/prepare.py --datasets alpaca oasst hh_rlhf

# Prepare everything except the huge datasets (C4, SmolLM)
python data/prepare.py --all

# Prepare everything including large datasets (needs ~200 GB disk + time)
python data/prepare.py --all --include-large

# Limit pretraining tokens (e.g. 100M tokens for quick experiments)
python data/prepare.py --datasets tinystories --max-tokens 100000000

# Use a different tokenizer
python data/prepare.py --datasets alpaca --tokenizer mistralai/Mistral-7B-v0.1

# Resume an interrupted run (skips datasets that already have meta.json)
python data/prepare.py --all --resume

# Show what is prepared so far
python data/prepare.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOKENIZER = "HuggingFaceTB/cosmo2-tokenizer"  # open-access, 49K vocab, matches SmolLM-Corpus
DEFAULT_OUTPUT_DIR = Path("/home/access/peter/trd/miniGPT/data")
DEFAULT_VAL_SPLIT = 0.005    # 0.5% of data for validation
DEFAULT_MAX_SEQ_LEN = 2048   # chunk size for pretraining .bin files

CHAT_TEMPLATE = (
    "<|system|>\nYou are a helpful assistant.\n"
    "<|user|>\n{instruction}\n"
    "<|assistant|>\n{response}"
)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    # ---- Pretraining -------------------------------------------------------
    "tinystories": {
        "kind": "pretrain",
        "hf_path": "roneneldan/TinyStories",
        "hf_split": "train",
        "text_field": "text",
        "description": "TinyStories — 2M short children's stories (~2B tokens). "
                       "Ideal for quick experiments and small-model testing.",
        "approx_size_gb": 2,
        "large": False,
    },
    "smollm": {
        "kind": "pretrain",
        "hf_path": "HuggingFaceTB/smollm-corpus",
        "hf_name": "fineweb-edu-dedup",
        "hf_split": "train",
        "text_field": "text",
        "description": "SmolLM-Corpus FineWeb-Edu-dedup — high-quality educational web text. "
                       "Primary pretraining corpus for the small/medium/large presets.",
        "approx_size_gb": 220,
        "large": True,
    },
    "c4": {
        "kind": "pretrain",
        "hf_path": "allenai/c4",
        "hf_name": "en",
        "hf_split": "train",
        "text_field": "text",
        "description": "C4 (Colossal Clean Crawled Corpus) — general web text. "
                       "Alternative pretraining corpus.",
        "approx_size_gb": 300,
        "large": True,
    },
    "openwebtext_10k": {
        "kind": "pretrain",
        "hf_path": "Skylion007/openwebtext",
        "hf_split": "train",
        "text_field": "text",
        "description": "OpenWebText (Skylion007) — Reddit-upvoted web text. "
                       "Use --max-tokens to cap size for quick experiments.",
        "approx_size_gb": 38,
        "large": False,
    },
    # ---- SFT ---------------------------------------------------------------
    "alpaca": {
        "kind": "sft",
        "hf_path": "tatsu-lab/alpaca",
        "hf_split": "train",
        "description": "Stanford Alpaca — 52K instruction-following examples. "
                       "Good starting point for SFT.",
        "approx_size_gb": 0.05,
        "large": False,
    },
    "oasst": {
        "kind": "sft",
        "hf_path": "OpenAssistant/oasst1",
        "hf_split": "train",
        "description": "OpenAssistant OASST1 — human-annotated multi-turn conversations.",
        "approx_size_gb": 0.1,
        "large": False,
    },
    "oasst2": {
        "kind": "sft",
        "hf_path": "OpenAssistant/oasst2",
        "hf_split": "train",
        "description": "OpenAssistant OASST2 — extended version of OASST1.",
        "approx_size_gb": 0.2,
        "large": False,
    },
    # ---- DPO ---------------------------------------------------------------
    "hh_rlhf": {
        "kind": "dpo",
        "hf_path": "Anthropic/hh-rlhf",
        "hf_split": "train",
        "description": "Anthropic HH-RLHF — preference pairs (chosen/rejected) "
                       "for DPO alignment training.",
        "approx_size_gb": 0.15,
        "large": False,
    },
}

SMALL_DATASETS = [k for k, v in DATASETS.items() if not v["large"]]
ALL_DATASETS = list(DATASETS.keys())


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Pretraining pipeline  →  packed uint16 .bin files
# ---------------------------------------------------------------------------

def _iter_pretrain_tokens(
    info: dict[str, Any],
    tokenizer,
    max_tokens: int | None,
) -> Iterator[list[int]]:
    """Stream and tokenize documents, yield token-ID lists."""
    from datasets import load_dataset

    ds = load_dataset(
        info["hf_path"],
        name=info.get("hf_name"),
        split=info["hf_split"],
        streaming=True,
    )

    total = 0
    for example in ds:
        text = example.get(info["text_field"], "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        yield ids
        total += len(ids)
        if max_tokens and total >= max_tokens:
            break


def _write_bin(path: Path, token_ids: list[int]) -> int:
    """Write a flat list of token IDs as uint16 binary. Returns byte count."""
    arr = np.array(token_ids, dtype=np.uint16)
    arr.tofile(str(path))
    return arr.nbytes


def prepare_pretrain(
    name: str,
    info: dict[str, Any],
    tokenizer,
    output_dir: Path,
    max_tokens: int | None,
    val_split: float,
    resume: bool,
) -> dict[str, Any]:
    """Download, tokenize, and pack a pretraining dataset into .bin files."""
    out = output_dir / name
    meta_path = out / "meta.json"

    if resume and meta_path.exists():
        logger.info("[%s] Already prepared — skipping (--resume)", name)
        with open(meta_path) as f:
            return json.load(f)

    out.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Starting pretraining preparation ...", name)
    t0 = time.time()

    # Collect all tokens into a flat list (streams, no full RAM load for small sets)
    all_ids: list[int] = []
    doc_count = 0
    for doc_ids in _iter_pretrain_tokens(info, tokenizer, max_tokens):
        all_ids.extend(doc_ids)
        doc_count += 1
        if doc_count % 10_000 == 0:
            logger.info("[%s] Processed %d docs, %d tokens ...", name, doc_count, len(all_ids))

    logger.info("[%s] Total: %d docs, %d tokens", name, doc_count, len(all_ids))

    # Train/val split
    n_val = max(1, int(len(all_ids) * val_split))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]

    # Write binary files
    train_bytes = _write_bin(out / "train.bin", train_ids)
    val_bytes = _write_bin(out / "val.bin", val_ids)

    elapsed = time.time() - t0
    meta = {
        "name": name,
        "kind": "pretrain",
        "tokenizer": tokenizer.name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "n_docs": doc_count,
        "n_train_tokens": len(train_ids),
        "n_val_tokens": len(val_ids),
        "train_size_mb": round(train_bytes / 1e6, 1),
        "val_size_mb": round(val_bytes / 1e6, 1),
        "dtype": "uint16",
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(elapsed, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "[%s] Done in %.0fs — train: %d tokens (%.1f MB), val: %d tokens (%.1f MB)",
        name, elapsed,
        len(train_ids), train_bytes / 1e6,
        len(val_ids), val_bytes / 1e6,
    )
    return meta


# ---------------------------------------------------------------------------
# SFT pipeline  →  .jsonl  with input_ids / labels (instruction tokens masked)
# ---------------------------------------------------------------------------

def _extract_sft_pair(example: dict, dataset_name: str) -> tuple[str, str] | None:
    """Return (instruction, response) or None if the example should be skipped."""
    if dataset_name == "alpaca":
        instruction = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        if inp:
            instruction = f"{instruction}\n\nInput: {inp}"
        response = example.get("output", "").strip()
    elif dataset_name in ("oasst", "oasst2"):
        # OASST stores individual messages; skip non-assistant messages
        role = example.get("role", "")
        if role != "assistant":
            return None
        # Try to get the parent instruction from the text field
        instruction = example.get("parent_id", "")  # use as placeholder
        response = example.get("text", "").strip()
        instruction = example.get("text", "").strip()  # fallback: use text as both
        # For OASST we use the text field as a completion target
        instruction = ""
        response = example.get("text", "").strip()
    else:
        instruction = example.get("instruction", example.get("prompt", "")).strip()
        response = example.get("response", example.get("output", example.get("completion", ""))).strip()

    if not response:
        return None
    return instruction, response


def prepare_sft(
    name: str,
    info: dict[str, Any],
    tokenizer,
    output_dir: Path,
    max_seq_len: int,
    val_split: float,
    resume: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    """Download and tokenize an SFT dataset into .jsonl files."""
    from datasets import load_dataset

    out = output_dir / name
    meta_path = out / "meta.json"

    if resume and meta_path.exists():
        logger.info("[%s] Already prepared — skipping (--resume)", name)
        with open(meta_path) as f:
            return json.load(f)

    out.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Starting SFT preparation ...", name)
    t0 = time.time()

    ds = load_dataset(
        info["hf_path"],
        name=info.get("hf_name"),
        split=info["hf_split"],
        streaming=True,
    )

    records: list[dict] = []
    skipped = 0

    for i, example in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        pair = _extract_sft_pair(example, name)
        if pair is None:
            skipped += 1
            continue
        instruction, response = pair

        # Tokenize full sequence
        if instruction:
            prompt_text = CHAT_TEMPLATE.format(instruction=instruction, response="")
            full_text = CHAT_TEMPLATE.format(instruction=instruction, response=response)
        else:
            # No instruction (e.g. OASST assistant turns) — treat as completion only
            full_text = response
            prompt_text = ""

        full_ids = tokenizer.encode(full_text, add_special_tokens=True)
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Mask instruction tokens
        if prompt_text:
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            n_mask = min(len(prompt_ids) - 1, len(labels))
            masked_labels = [-100] * n_mask + labels[n_mask:]
        else:
            masked_labels = labels

        records.append({
            "input_ids": input_ids,
            "labels": masked_labels,
            "attention_mask": [1] * len(input_ids),
            "instruction": instruction[:200] if instruction else "",  # truncated for reference
            "response_len": len(labels) - masked_labels.count(-100),
        })

        if len(records) % 5_000 == 0:
            logger.info("[%s] Processed %d samples ...", name, len(records))

    logger.info("[%s] Total: %d samples (%d skipped)", name, len(records), skipped)

    # Shuffle and split
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(records)).tolist()
    n_val = max(1, int(len(records) * val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    def write_jsonl(path: Path, idxs: list[int]) -> None:
        with open(path, "w") as f:
            for idx in idxs:
                f.write(json.dumps(records[idx]) + "\n")

    write_jsonl(out / "train.jsonl", train_indices)
    write_jsonl(out / "val.jsonl", val_indices)

    avg_len = sum(len(r["input_ids"]) for r in records) / max(len(records), 1)
    elapsed = time.time() - t0

    meta = {
        "name": name,
        "kind": "sft",
        "tokenizer": tokenizer.name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_skipped": skipped,
        "avg_seq_len": round(avg_len, 1),
        "max_seq_len": max_seq_len,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(elapsed, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "[%s] Done in %.0fs — train: %d samples, val: %d samples, avg_len: %.0f",
        name, elapsed, len(train_indices), len(val_indices), avg_len,
    )
    return meta


# ---------------------------------------------------------------------------
# DPO pipeline  →  .jsonl  with prompt / chosen / rejected
# ---------------------------------------------------------------------------

def prepare_dpo(
    name: str,
    info: dict[str, Any],
    output_dir: Path,
    val_split: float,
    resume: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    """Download and structure a DPO preference dataset into .jsonl files."""
    from datasets import load_dataset

    out = output_dir / name
    meta_path = out / "meta.json"

    if resume and meta_path.exists():
        logger.info("[%s] Already prepared — skipping (--resume)", name)
        with open(meta_path) as f:
            return json.load(f)

    out.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Starting DPO preparation ...", name)
    t0 = time.time()

    ds = load_dataset(
        info["hf_path"],
        name=info.get("hf_name"),
        split=info["hf_split"],
        streaming=True,
    )

    records: list[dict] = []
    skipped = 0

    for i, example in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        chosen = example.get("chosen", "").strip()
        rejected = example.get("rejected", "").strip()
        if not chosen or not rejected:
            skipped += 1
            continue

        # Parse HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ..."
        if "\n\nAssistant:" in chosen:
            parts = chosen.rsplit("\n\nAssistant:", 1)
            prompt = parts[0].strip()
            chosen_response = parts[1].strip()
        else:
            prompt = ""
            chosen_response = chosen

        if "\n\nAssistant:" in rejected:
            parts = rejected.rsplit("\n\nAssistant:", 1)
            rejected_response = parts[1].strip()
        else:
            rejected_response = rejected

        records.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })

        if len(records) % 5_000 == 0:
            logger.info("[%s] Processed %d samples ...", name, len(records))

    logger.info("[%s] Total: %d pairs (%d skipped)", name, len(records), skipped)

    # Shuffle and split
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(records)).tolist()
    n_val = max(1, int(len(records) * val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    def write_jsonl(path: Path, idxs: list[int]) -> None:
        with open(path, "w") as f:
            for idx in idxs:
                f.write(json.dumps(records[idx]) + "\n")

    write_jsonl(out / "train.jsonl", train_indices)
    write_jsonl(out / "val.jsonl", val_indices)

    elapsed = time.time() - t0
    meta = {
        "name": name,
        "kind": "dpo",
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_skipped": skipped,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(elapsed, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "[%s] Done in %.0fs — train: %d pairs, val: %d pairs",
        name, elapsed, len(train_indices), len(val_indices),
    )
    return meta


# ---------------------------------------------------------------------------
# Validation: spot-check prepared data
# ---------------------------------------------------------------------------

def validate(name: str, output_dir: Path) -> bool:
    """Spot-check a prepared dataset. Returns True if looks OK."""
    out = output_dir / name
    meta_path = out / "meta.json"

    if not meta_path.exists():
        logger.warning("[%s] meta.json missing — not prepared yet", name)
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    kind = meta["kind"]

    if kind == "pretrain":
        train_bin = out / "train.bin"
        val_bin = out / "val.bin"
        if not train_bin.exists() or not val_bin.exists():
            logger.error("[%s] Missing .bin files", name)
            return False

        # Load first few tokens and check dtype/shape
        arr = np.fromfile(str(train_bin), dtype=np.uint16)
        if len(arr) == 0:
            logger.error("[%s] train.bin is empty", name)
            return False

        logger.info("[%s] OK — %d train tokens, %d val tokens (uint16 .bin)",
                    name, meta["n_train_tokens"], meta["n_val_tokens"])
        return True

    else:  # sft or dpo
        train_jl = out / "train.jsonl"
        val_jl = out / "val.jsonl"
        if not train_jl.exists() or not val_jl.exists():
            logger.error("[%s] Missing .jsonl files", name)
            return False

        # Read first line and validate fields
        with open(train_jl) as f:
            first_line = f.readline()
        if not first_line.strip():
            logger.error("[%s] train.jsonl is empty", name)
            return False

        record = json.loads(first_line)
        if kind == "sft":
            required = {"input_ids", "labels", "attention_mask"}
        else:
            required = {"prompt", "chosen", "rejected"}

        missing = required - set(record.keys())
        if missing:
            logger.error("[%s] Missing fields: %s", name, missing)
            return False

        if kind == "sft":
            n_train = meta["n_train"]
            n_val = meta["n_val"]
            logger.info("[%s] OK — %d train samples, %d val samples (avg_len=%.0f, .jsonl)",
                        name, n_train, n_val, meta.get("avg_seq_len", 0))
        else:
            logger.info("[%s] OK — %d train pairs, %d val pairs (.jsonl)",
                        name, meta["n_train"], meta["n_val"])
        return True


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def print_status(output_dir: Path) -> None:
    """Print a table of all datasets and their preparation status."""
    print(f"\n{'Dataset':<15} {'Kind':<8} {'Status':<12} {'Size / Tokens':<25} {'Description'}")
    print("-" * 100)
    for name, info in DATASETS.items():
        meta_path = output_dir / name / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if info["kind"] == "pretrain":
                size_str = f"{meta['n_train_tokens']:,} train tokens"
            else:
                size_str = f"{meta['n_train']:,} train / {meta['n_val']:,} val"
            status = "READY"
        else:
            size_str = f"~{info['approx_size_gb']} GB download"
            status = "not prepared"

        large_tag = " [LARGE]" if info["large"] else ""
        print(f"{name:<15} {info['kind']:<8} {status:<12} {size_str:<25} {info['description'][:60]}{large_tag}")
    print()


# ---------------------------------------------------------------------------
# DataLoader helpers for training — load from prepared data/
# ---------------------------------------------------------------------------

class BinDataset:
    """Memory-mapped pretraining dataset from a prepared .bin file.

    Returns overlapping chunks of ``seq_len + 1`` tokens for next-token
    prediction. Usable as a PyTorch Dataset.

    Parameters
    ----------
    bin_path:
        Path to the prepared ``train.bin`` or ``val.bin`` file.
    seq_len:
        Sequence length (model context window).
    """

    def __init__(self, bin_path: str | Path, seq_len: int = 2048) -> None:
        import torch
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int):
        import torch
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


class JsonlSFTDataset:
    """SFT dataset from a prepared ``train.jsonl`` or ``val.jsonl`` file.

    Parameters
    ----------
    jsonl_path:
        Path to the prepared ``.jsonl`` file.
    max_seq_len:
        Hard cap on sequence length.
    """

    def __init__(self, jsonl_path: str | Path, max_seq_len: int = 2048) -> None:
        self.records: list[dict] = []
        self.max_seq_len = max_seq_len
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import torch
        r = self.records[idx]
        ids = r["input_ids"][: self.max_seq_len]
        lbs = r["labels"][: self.max_seq_len]
        mask = r["attention_mask"][: self.max_seq_len]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lbs, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def build_pretrain_dataloader_from_disk(
    dataset_name: str,
    data_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split: str = "train",
    seq_len: int = 2048,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """Build a DataLoader from a prepared pretraining .bin file.

    Falls back to streaming from HuggingFace if the .bin file doesn't exist.
    """
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    bin_path = Path(data_dir) / dataset_name / f"{split}.bin"
    if not bin_path.exists():
        logger.warning(
            "[%s] %s not found — falling back to HuggingFace streaming. "
            "Run `python data/prepare.py --datasets %s` to prepare locally.",
            dataset_name, bin_path, dataset_name,
        )
        from training.data import build_pretraining_dataloader
        return build_pretraining_dataloader(
            dataset_name=dataset_name,
            max_seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    dataset = BinDataset(bin_path, seq_len=seq_len)
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_sft_dataloader_from_disk(
    dataset_name: str,
    data_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split: str = "train",
    max_seq_len: int = 2048,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
):
    """Build a DataLoader from a prepared SFT .jsonl file.

    Falls back to streaming from HuggingFace if not prepared locally.
    """
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    jsonl_path = Path(data_dir) / dataset_name / f"{split}.jsonl"
    if not jsonl_path.exists():
        logger.warning(
            "[%s] %s not found — falling back to HuggingFace streaming. "
            "Run `python data/prepare.py --datasets %s` to prepare locally.",
            dataset_name, jsonl_path, dataset_name,
        )
        from training.data import build_sft_dataloader
        return build_sft_dataloader(
            dataset_name=dataset_name,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    dataset = JsonlSFTDataset(jsonl_path, max_seq_len=max_seq_len)
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def collate(batch):
        import torch
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            l = b["input_ids"].size(0)
            input_ids[i, :l] = b["input_ids"]
            labels[i, :l] = b["labels"]
            attention_mask[i, :l] = b["attention_mask"]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cfg():
    """Load central config.yaml via OmegaConf (silent fallback if missing)."""
    try:
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        return OmegaConf.load(cfg_path)
    except Exception:
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load central config — provides defaults for all arguments below
    cfg = _load_cfg()
    _d = cfg.data if cfg else None
    _p = cfg.paths if cfg else None

    _def_output     = str(_p.data_dir)    if _p else str(DEFAULT_OUTPUT_DIR)
    _def_tokenizer  = str(_d.tokenizer)   if _d else DEFAULT_TOKENIZER
    _def_max_tok    = int(_d.max_tokens)  if _d and _d.max_tokens  else None
    _def_max_samp   = int(_d.max_samples) if _d and _d.max_samples else None
    _def_seq_len    = int(_d.max_seq_len) if _d else DEFAULT_MAX_SEQ_LEN
    _def_val_split  = float(_d.val_split) if _d else DEFAULT_VAL_SPLIT
    _def_datasets   = list(_d.datasets)   if _d else None

    parser = argparse.ArgumentParser(
        description="Download and prepare MiniGPT training datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        metavar="NAME",
        help=f"One or more dataset names (space or comma-separated). Available: {', '.join(ALL_DATASETS)}",
    )
    parser.add_argument("--all", action="store_true",
                        help="Prepare all datasets (skips large ones unless --include-large)")
    parser.add_argument("--include-large", action="store_true",
                        help="Include large datasets (C4, SmolLM-Corpus) when using --all")
    parser.add_argument("--output-dir", type=str, default=_def_output,
                        help="Root output directory")
    parser.add_argument("--tokenizer", type=str, default=_def_tokenizer,
                        help="HuggingFace tokenizer name")
    parser.add_argument("--max-tokens", type=int, default=_def_max_tok,
                        help="Max tokens per pretraining dataset (None = full dataset)")
    parser.add_argument("--max-samples", type=int, default=_def_max_samp,
                        help="Max samples per SFT/DPO dataset (None = full dataset)")
    parser.add_argument("--max-seq-len", type=int, default=_def_seq_len,
                        help="Sequence length for SFT tokenization")
    parser.add_argument("--val-split", type=float, default=_def_val_split,
                        help="Fraction of data held out for validation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip datasets that already have meta.json")
    parser.add_argument("--validate", action="store_true",
                        help="Validate prepared datasets and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show preparation status of all datasets and exit")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets and exit")
    args = parser.parse_args()

    # Fall back to config.yaml datasets when no CLI datasets provided
    if not args.datasets and not args.all and _def_datasets:
        args.datasets = _def_datasets

    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Info-only modes
    # ------------------------------------------------------------------
    if args.list:
        print(f"\n{'Name':<15} {'Kind':<10} {'Large':<8} Description")
        print("-" * 80)
        for name, info in DATASETS.items():
            large = "yes" if info["large"] else ""
            print(f"{name:<15} {info['kind']:<10} {large:<8} {info['description'][:60]}")
        print()
        return

    if args.status:
        print_status(output_dir)
        return

    if args.validate:
        all_ok = True
        for name in ALL_DATASETS:
            ok = validate(name, output_dir)
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    # ------------------------------------------------------------------
    # Resolve which datasets to prepare
    # ------------------------------------------------------------------
    if args.all:
        targets = [n for n, v in DATASETS.items()
                   if not v["large"] or args.include_large]
    elif args.datasets:
        # Accept both space-separated (nargs="+") and comma-separated values
        targets = [n.strip() for token in args.datasets for n in token.split(",") if n.strip()]
        unknown = [n for n in targets if n not in DATASETS]
        if unknown:
            logger.error("Unknown datasets: %s. Run --list to see available.", unknown)
            sys.exit(1)
    else:
        parser.print_help()
        print("\nTip: run with --list to see available datasets, --status to check prepared state.")
        sys.exit(0)

    if not targets:
        logger.error("No datasets selected.")
        sys.exit(1)

    logger.info("Datasets to prepare: %s", targets)
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("Tokenizer: %s", args.tokenizer)

    # Load tokenizer once (shared across all SFT datasets)
    tokenizer = None
    pretrain_targets = [n for n in targets if DATASETS[n]["kind"] == "pretrain"]
    sft_dpo_targets  = [n for n in targets if DATASETS[n]["kind"] in ("sft", "dpo")]

    if pretrain_targets or sft_dpo_targets:
        logger.info("Loading tokenizer %s ...", args.tokenizer)
        tokenizer = load_tokenizer(args.tokenizer)
        logger.info("Tokenizer loaded. Vocab size: %d", tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # Prepare each dataset
    # ------------------------------------------------------------------
    results: list[dict] = []
    failed: list[str] = []

    for name in targets:
        info = DATASETS[name]
        try:
            if info["kind"] == "pretrain":
                meta = prepare_pretrain(
                    name=name,
                    info=info,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    max_tokens=args.max_tokens,
                    val_split=args.val_split,
                    resume=args.resume,
                )
            elif info["kind"] == "sft":
                meta = prepare_sft(
                    name=name,
                    info=info,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    max_seq_len=args.max_seq_len,
                    val_split=args.val_split,
                    resume=args.resume,
                    max_samples=args.max_samples,
                )
            elif info["kind"] == "dpo":
                meta = prepare_dpo(
                    name=name,
                    info=info,
                    output_dir=output_dir,
                    val_split=args.val_split,
                    resume=args.resume,
                    max_samples=args.max_samples,
                )
            results.append(meta)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            break
        except Exception as exc:
            logger.error("[%s] FAILED: %s", name, exc, exc_info=True)
            failed.append(name)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)
    for meta in results:
        n = meta["name"]
        k = meta["kind"]
        if k == "pretrain":
            print(f"  {n:<15} {k:<8}  {meta['n_train_tokens']:>12,} train tokens   "
                  f"{meta['n_val_tokens']:>10,} val tokens   "
                  f"({meta['train_size_mb']:.0f} MB train)")
        else:
            print(f"  {n:<15} {k:<8}  {meta['n_train']:>12,} train samples  "
                  f"{meta['n_val']:>10,} val samples")
    if failed:
        print(f"\n  FAILED: {failed}")
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
