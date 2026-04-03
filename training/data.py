"""Dataset loading for MiniGPT pretraining and fine-tuning.

Supports streaming from HuggingFace datasets:
- SmolLM-Corpus (FineWeb-Edu-dedup subset) for pretraining
- TinyStories for small-scale experiments
- C4 for general pretraining
- OpenAssistant (OASST) for SFT
- Alpaca for SFT
- Anthropic HH-RLHF for DPO

Includes a custom data collator for causal language modelling with
proper padding, masking, and label shifting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# Default tokenizer to reuse (Llama 3.2 1B)
DEFAULT_TOKENIZER = "meta-llama/Llama-3.2-1B"

# Dataset identifiers on HuggingFace Hub
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "smollm": {
        "path": "HuggingFaceTB/smollm-corpus",
        "name": "fineweb-edu-dedup",
        "split": "train",
        "text_field": "text",
    },
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "split": "train",
        "text_field": "text",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "train",
        "text_field": "text",
    },
    "oasst": {
        "path": "OpenAssistant/oasst1",
        "split": "train",
        "text_field": "text",
    },
    "oasst2": {
        "path": "OpenAssistant/oasst2",
        "split": "train",
        "text_field": "text",
    },
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "split": "train",
        "text_field": "text",
    },
    "hh_rlhf": {
        "path": "Anthropic/hh-rlhf",
        "split": "train",
        "text_field": "chosen",
    },
    "c4_val": {
        "path": "allenai/c4",
        "name": "en",
        "split": "validation",
        "text_field": "text",
    },
}


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_name: str = DEFAULT_TOKENIZER):
    """Load a HuggingFace tokenizer, setting pad_token if missing."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# Streaming dataset wrapper
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """Wraps a HuggingFace streaming dataset for causal LM pretraining.

    Tokenizes documents on-the-fly and packs them into fixed-length
    chunks of ``max_seq_len`` tokens, concatenating documents with an
    EOS separator to avoid wasting partial-sequence padding.

    Parameters
    ----------
    dataset_name:
        Key into ``DATASET_REGISTRY`` or a HuggingFace dataset path.
    tokenizer_name:
        HuggingFace tokenizer identifier.
    max_seq_len:
        Maximum sequence length per sample.
    max_tokens:
        Optional cap on total tokens streamed (for controlling data budget).
    """

    def __init__(
        self,
        dataset_name: str = "smollm",
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_seq_len: int = 2048,
        max_tokens: int | None = None,
        dataset_config_name: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.dataset_config_name = dataset_config_name

    def _get_dataset_info(self) -> dict[str, Any]:
        """Resolve dataset info from registry or use as raw HF path."""
        if self.dataset_name in DATASET_REGISTRY:
            info = dict(DATASET_REGISTRY[self.dataset_name])
        else:
            info = {
                "path": self.dataset_name,
                "split": "train",
                "text_field": "text",
            }
        if self.dataset_config_name is not None:
            info["name"] = self.dataset_config_name
        return info

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset

        info = self._get_dataset_info()
        tokenizer = load_tokenizer(self.tokenizer_name)

        ds = load_dataset(
            info["path"],
            name=info.get("name"),
            split=info["split"],
            streaming=True,
            
        )

        text_field = info["text_field"]
        buffer: list[int] = []
        total_tokens = 0

        for example in ds:
            text = example.get(text_field, "")
            if not text:
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids.append(tokenizer.eos_token_id)
            buffer.extend(token_ids)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                total_tokens += self.max_seq_len
                if self.max_tokens and total_tokens > self.max_tokens:
                    return

                yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

class SFTDataset(IterableDataset):
    """Instruction-following dataset for supervised fine-tuning.

    Formats examples with a chat template and masks instruction tokens
    so that loss is only computed on response tokens.

    Parameters
    ----------
    dataset_name:
        One of ``"oasst"``, ``"oasst2"``, ``"alpaca"``, or a HF path.
    tokenizer_name:
        HuggingFace tokenizer identifier.
    max_seq_len:
        Maximum sequence length.
    """

    CHAT_TEMPLATE = (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\n{instruction}\n"
        "<|assistant|>\n{response}"
    )

    def __init__(
        self,
        dataset_name: str = "alpaca",
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len

    @staticmethod
    def _extract_instruction_response(example: dict[str, Any], dataset_name: str) -> tuple[str, str]:
        """Extract instruction and response from different dataset formats."""
        if dataset_name == "alpaca":
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            if inp:
                instruction = f"{instruction}\n\nInput: {inp}"
            response = example.get("output", "")
        elif dataset_name in ("oasst", "oasst2"):
            # OASST stores conversation trees; we extract first turn pairs.
            instruction = example.get("instruction", example.get("text", ""))
            response = example.get("response", "")
        else:
            instruction = example.get("instruction", example.get("prompt", ""))
            response = example.get("response", example.get("output", example.get("completion", "")))
        return instruction, response

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset

        info = DATASET_REGISTRY.get(self.dataset_name, {"path": self.dataset_name, "split": "train"})
        tokenizer = load_tokenizer(self.tokenizer_name)

        ds = load_dataset(
            info["path"],
            name=info.get("name"),
            split=info.get("split", "train"),
            streaming=True,
            
        )

        for example in ds:
            instruction, response = self._extract_instruction_response(example, self.dataset_name)
            if not instruction or not response:
                continue

            # Format with chat template
            prompt_text = self.CHAT_TEMPLATE.format(instruction=instruction, response="")
            full_text = self.CHAT_TEMPLATE.format(instruction=instruction, response=response)

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)

            if len(full_ids) > self.max_seq_len:
                full_ids = full_ids[: self.max_seq_len]

            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            labels = torch.tensor(full_ids[1:], dtype=torch.long)

            # Mask instruction tokens (set to -100 so they are ignored in loss)
            n_prompt = len(prompt_ids) - 1  # -1 because labels are shifted
            if n_prompt > 0:
                labels[:n_prompt] = -100

            yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# DPO dataset
# ---------------------------------------------------------------------------

class DPODataset(IterableDataset):
    """Preference dataset for Direct Preference Optimization.

    Yields (prompt, chosen, rejected) triples from Anthropic HH-RLHF
    or compatible datasets.
    """

    def __init__(
        self,
        dataset_name: str = "hh_rlhf",
        tokenizer_name: str = DEFAULT_TOKENIZER,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        info = DATASET_REGISTRY.get(self.dataset_name, {"path": self.dataset_name, "split": "train"})

        ds = load_dataset(
            info["path"],
            name=info.get("name"),
            split=info.get("split", "train"),
            streaming=True,
            
        )

        for example in ds:
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")
            if not chosen or not rejected:
                continue

            # Extract the last assistant turn as prompt context
            # HH-RLHF format: "\n\nHuman: ... \n\nAssistant: ..."
            prompt = ""
            if "\n\nAssistant:" in chosen:
                parts = chosen.rsplit("\n\nAssistant:", 1)
                prompt = parts[0]
                chosen_response = parts[1].strip() if len(parts) > 1 else chosen
            else:
                chosen_response = chosen

            if "\n\nAssistant:" in rejected:
                parts = rejected.rsplit("\n\nAssistant:", 1)
                rejected_response = parts[1].strip() if len(parts) > 1 else rejected
            else:
                rejected_response = rejected

            yield {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            }


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class CausalLMCollator:
    """Custom data collator for causal language modelling.

    Pads sequences to the longest in the batch (or ``max_seq_len``),
    creates attention masks, and shifts labels so the model predicts
    the next token.

    Parameters
    ----------
    pad_token_id:
        Token ID used for padding.
    max_seq_len:
        Hard cap on sequence length.
    """

    pad_token_id: int = 0
    max_seq_len: int = 2048

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids_list = [ex["input_ids"][: self.max_seq_len] for ex in batch]
        labels_list = [ex["labels"][: self.max_seq_len] for ex in batch]

        max_len = max(ids.size(0) for ids in input_ids_list)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            seq_len = input_ids.size(0)
            pad_len = max_len - seq_len

            padded_input_ids.append(
                torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            )
            attention_masks.append(
                torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_pretraining_dataloader(
    dataset_name: str = "smollm",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_seq_len: int = 2048,
    batch_size: int = 8,
    max_tokens: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for pretraining.

    Parameters
    ----------
    dataset_name:
        Key in ``DATASET_REGISTRY`` or HuggingFace dataset path.
    tokenizer_name:
        Tokenizer to use for encoding text.
    max_seq_len:
        Maximum token sequence length.
    batch_size:
        Batch size.
    max_tokens:
        Optional token budget cap.
    num_workers:
        Number of dataloader workers.

    Returns
    -------
    DataLoader
    """
    tokenizer = load_tokenizer(tokenizer_name)
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        max_tokens=max_tokens,
    )
    collator = CausalLMCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        max_seq_len=max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_sft_dataloader(
    dataset_name: str = "alpaca",
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for supervised fine-tuning."""
    tokenizer = load_tokenizer(tokenizer_name)
    dataset = SFTDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
    )
    collator = CausalLMCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        max_seq_len=max_seq_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
