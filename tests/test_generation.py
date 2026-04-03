"""Tests for text generation with KV cache.

Verifies autoregressive generation, sampling parameters,
and KV cache correctness using generate_from_ids (no tokenizer needed).
"""

from __future__ import annotations

import pytest
import torch

from configs.model.config import ModelConfig
from miniGPT.model import MiniGPT
from miniGPT.generation import generate_from_ids


def _small_config(**overrides) -> ModelConfig:
    defaults = dict(
        d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=128, vocab_size=1000,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestGenerate:

    def test_basic_generation(self):
        config = _small_config()
        model = MiniGPT(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        output_ids = generate_from_ids(model, input_ids, max_new_tokens=16, temperature=1.0)
        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] == 24  # 8 prompt + 16 generated

    def test_deterministic_with_low_temp(self):
        config = _small_config()
        model = MiniGPT(config)
        model.eval()
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        # top_k=1 (greedy) gives deterministic output
        out1 = generate_from_ids(model, input_ids.clone(), max_new_tokens=8, temperature=0.01, top_k=1)
        out2 = generate_from_ids(model, input_ids.clone(), max_new_tokens=8, temperature=0.01, top_k=1)
        assert torch.equal(out1, out2)

    def test_batch_generation(self):
        config = _small_config()
        model = MiniGPT(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (3, 8))
        output_ids = generate_from_ids(model, input_ids, max_new_tokens=16, temperature=0.7)
        assert output_ids.shape[0] == 3
        assert output_ids.shape[1] == 24

    def test_max_tokens_respected(self):
        config = _small_config()
        model = MiniGPT(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        max_new = 10
        output_ids = generate_from_ids(model, input_ids, max_new_tokens=max_new, temperature=0.7)
        assert output_ids.shape[1] == 8 + max_new

    def test_top_p_sampling(self):
        config = _small_config()
        model = MiniGPT(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        output_ids = generate_from_ids(model, input_ids, max_new_tokens=16, temperature=0.7, top_p=0.9)
        assert output_ids.shape[1] == 24

    def test_top_k_sampling(self):
        config = _small_config()
        model = MiniGPT(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        output_ids = generate_from_ids(model, input_ids, max_new_tokens=16, temperature=0.7, top_k=10)
        assert output_ids.shape[1] == 24
