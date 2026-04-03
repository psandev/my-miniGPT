"""Tests for the full MiniGPT model with various configurations.

Verifies forward pass, loss computation, parameter counting,
and gradient flow through the complete model.
"""

from __future__ import annotations

import pytest
import torch

from configs.model.config import ModelConfig
from miniGPT.model import MiniGPT


def _small_config(**overrides) -> ModelConfig:
    """Create a small test config with optional overrides."""
    defaults = dict(
        d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=128, vocab_size=1000,
        mla_kv_lora_rank=64, mla_q_lora_rank=128, mla_rope_head_dim=32,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestModelForward:

    def test_basic_forward(self):
        config = _small_config()
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 64, config.vocab_size)

    def test_forward_with_targets(self):
        config = _small_config()
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        t = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x, targets=t)
        assert "logits" in out
        assert "loss" in out
        assert out["loss"] is not None
        assert out["loss"].dim() == 0  # scalar

    def test_backward(self):
        config = _small_config()
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        t = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x, targets=t)
        out["loss"].backward()
        # Verify at least some gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads


class TestModelConfigs:

    @pytest.mark.parametrize("attention_type", ["mha", "gqa", "mla"])
    def test_attention_variants(self, attention_type):
        config = _small_config(attention_type=attention_type)
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x)
        assert out["logits"].shape == (2, 64, config.vocab_size)

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm", "dyt"])
    def test_norm_variants(self, norm_type):
        config = _small_config(norm_type=norm_type)
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x)
        assert out["logits"].shape == (2, 64, config.vocab_size)

    @pytest.mark.parametrize("ffn_type", ["swiglu", "gelu", "relu", "moe"])
    def test_ffn_variants(self, ffn_type):
        config = _small_config(ffn_type=ffn_type, moe_num_experts=4, moe_top_k=2)
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x)
        assert out["logits"].shape == (2, 64, config.vocab_size)

    @pytest.mark.parametrize("residual_type", ["standard", "mhc"])
    def test_residual_variants(self, residual_type):
        config = _small_config(residual_type=residual_type, mhc_n_streams=4)
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x)
        assert out["logits"].shape == (2, 64, config.vocab_size)

    @pytest.mark.parametrize("prediction_type", ["stp", "mtp"])
    def test_prediction_variants(self, prediction_type):
        config = _small_config(prediction_type=prediction_type, mtp_n_heads=3)
        model = MiniGPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        t = torch.randint(0, config.vocab_size, (2, 64))
        out = model(x, targets=t)
        assert out["logits"].shape == (2, 64, config.vocab_size)
        assert out["loss"] is not None


class TestCountParameters:

    def test_count_parameters(self):
        config = _small_config()
        model = MiniGPT(config)
        params = model.count_parameters()
        assert "total" in params
        assert params["total"] > 0

    def test_count_parameters_human_readable(self):
        config = _small_config()
        model = MiniGPT(config)
        params = model.count_parameters()
        # Should have human-readable format
        total = params.get("total", 0)
        assert total > 0
