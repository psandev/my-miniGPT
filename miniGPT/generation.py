"""
Autoregressive text generation with KV cache for MiniGPT.

Features
--------
* KV cache for efficient token-by-token inference.
* Top-k sampling (Fan et al., 2018).
* Top-p / nucleus sampling (Holtzman et al., 2019).
* Temperature scaling.
* Repetition penalty (Keskar et al., 2019).
* Streaming output via Python generator (``yield`` tokens as produced).
* Batch generation support.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from miniGPT.model import MiniGPT


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalise tokens that have already been generated.

    For each token in *generated_ids*, divide its logit by *penalty* if
    positive, or multiply by *penalty* if negative.
    """
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, generated_ids)
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits.scatter_(1, generated_ids, score)
    return logits


def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero-out logits outside the top-k values."""
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Shift right so the first token above threshold is kept
    mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
    sorted_logits[mask] = float("-inf")
    # Scatter back
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


@torch.inference_mode()
def generate(
    model: MiniGPT,
    tokenizer,
    prompt: str | list[str],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    stream: bool = False,
    device: str | torch.device = "cpu",
) -> str | list[str] | Iterator[str]:
    """Generate text autoregressively from a prompt.

    Parameters
    ----------
    model : MiniGPT
        The language model.
    tokenizer
        Any HuggingFace-compatible tokenizer (must support ``encode`` /
        ``decode``).
    prompt : str or list[str]
        One or more prompts.  If a list is provided, batch generation is used.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature.  ``1.0`` = unchanged, ``< 1`` = sharper,
        ``> 1`` = flatter.
    top_k : int
        Keep only the top-k logits before sampling.  ``0`` = disabled.
    top_p : float
        Nucleus sampling threshold.  ``1.0`` = disabled.
    repetition_penalty : float
        Penalty factor for repeated tokens.  ``1.0`` = disabled.
    stream : bool
        If *True*, return an iterator that yields tokens as they are
        generated (single-prompt only).
    device : str or torch.device
        Device to run generation on.

    Returns
    -------
    str, list[str], or Iterator[str]
        Generated text.  If *stream* is True, yields token strings.
    """
    model.eval()

    # Handle batched input
    is_batch = isinstance(prompt, list)
    prompts = prompt if is_batch else [prompt]

    # Tokenise
    encodings = [tokenizer.encode(p) for p in prompts]
    max_prompt_len = max(len(e) for e in encodings)

    # Pad to equal length (left-padding for causal generation)
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
    padded = []
    for enc in encodings:
        padding = [pad_id] * (max_prompt_len - len(enc))
        padded.append(padding + enc)

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    generated = input_ids.clone()

    kv_caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # Prefill: process the full prompt
    positions = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand(input_ids.size(0), -1)
    logits, kv_caches = model.forward_with_cache(
        input_ids, kv_caches=None, positions=positions
    )

    def _step(step_logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits (B, V)."""
        if temperature != 1.0:
            step_logits = step_logits / temperature
        step_logits = _apply_repetition_penalty(step_logits, generated, repetition_penalty)
        step_logits = _top_k_filtering(step_logits, top_k)
        step_logits = _top_p_filtering(step_logits, top_p)
        probs = F.softmax(step_logits, dim=-1)
        if temperature == 0:
            return probs.argmax(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)

    def _generate_tokens() -> Iterator[str]:
        nonlocal kv_caches, generated, logits

        for i in range(max_new_tokens):
            # Get logits for the last position
            next_logits = logits[:, -1, :]  # (B, V)
            next_token = _step(next_logits)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and (next_token == eos_id).all():
                break

            # Yield decoded token (for streaming, single prompt only)
            if stream and not is_batch:
                yield tokenizer.decode(next_token[0].tolist())

            # Next step: feed only the new token
            cur_pos = generated.size(1) - 1
            pos = torch.tensor([[cur_pos]], device=device).expand(next_token.size(0), -1)
            logits, kv_caches = model.forward_with_cache(
                next_token, kv_caches=kv_caches, positions=pos
            )

    if stream and not is_batch:
        return _generate_tokens()

    # Non-streaming: consume the full generator
    for _ in _generate_tokens():
        pass

    # Decode each sequence, stripping the prompt
    results = []
    for i, enc in enumerate(encodings):
        prompt_len = max_prompt_len  # includes padding
        output_ids = generated[i, prompt_len:].tolist()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and eos_id in output_ids:
            output_ids = output_ids[: output_ids.index(eos_id)]
        results.append(tokenizer.decode(output_ids))

    return results if is_batch else results[0]


@torch.inference_mode()
def generate_from_ids(
    model: "MiniGPT",
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Generate tokens from a tensor of input IDs (no tokenizer required).

    Parameters
    ----------
    model : MiniGPT
    input_ids : LongTensor of shape ``(B, S)``
    max_new_tokens : int
    temperature, top_k, top_p, repetition_penalty : sampling controls

    Returns
    -------
    LongTensor of shape ``(B, S + max_new_tokens)``
    """
    model.eval()
    device = input_ids.device
    generated = input_ids.clone()

    positions = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand(input_ids.size(0), -1)
    logits, kv_caches = model.forward_with_cache(input_ids, kv_caches=None, positions=positions)

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        if temperature != 1.0 and temperature > 0:
            next_logits = next_logits / temperature
        next_logits = _apply_repetition_penalty(next_logits, generated, repetition_penalty)
        next_logits = _top_k_filtering(next_logits, top_k)
        next_logits = _top_p_filtering(next_logits, top_p)
        probs = F.softmax(next_logits, dim=-1)
        if temperature == 0 or top_k == 1:
            next_token = probs.argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        cur_pos = generated.size(1) - 1
        pos = torch.tensor([[cur_pos]], device=device).expand(next_token.size(0), -1)
        logits, kv_caches = model.forward_with_cache(next_token, kv_caches=kv_caches, positions=pos)

    return generated
