"""Tests for all attention variants: MHA, GQA, MLA.

Verifies forward pass output shapes, backward gradient flow,
and correct handling of different head configurations.
"""

from __future__ import annotations

import pytest
import torch

from configs.model.config import ModelConfig
from miniGPT.modules.attention import build_attention


def _forward(attn, x, **kwargs):
    """Unpack attention output — returns (tensor, kv_cache) tuple."""
    result = attn(x, **kwargs)
    if isinstance(result, tuple):
        return result[0]
    return result


@pytest.fixture
def base_config():
    return ModelConfig(d_model=256, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128)


class TestMHA:
    """Tests for Multi-Head Attention (n_kv_heads == n_heads)."""

    def test_forward_shape(self):
        config = ModelConfig(d_model=256, n_layers=2, n_heads=4, n_kv_heads=4, attention_type="mha")
        attn = build_attention(config)
        x = torch.randn(2, 64, 256)
        out = _forward(attn, x)
        assert out.shape == (2, 64, 256)

    def test_backward(self):
        config = ModelConfig(d_model=256, n_layers=2, n_heads=4, n_kv_heads=4, attention_type="mha")
        attn = build_attention(config)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = _forward(attn, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 64, 256)


class TestGQA:
    """Tests for Grouped Query Attention (n_kv_heads < n_heads)."""

    def test_forward_shape(self):
        config = ModelConfig(d_model=256, n_layers=2, n_heads=4, n_kv_heads=2, attention_type="gqa")
        attn = build_attention(config)
        x = torch.randn(2, 64, 256)
        out = _forward(attn, x)
        assert out.shape == (2, 64, 256)

    def test_backward(self):
        config = ModelConfig(d_model=256, n_layers=2, n_heads=4, n_kv_heads=2, attention_type="gqa")
        attn = build_attention(config)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = _forward(attn, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_mqa(self):
        """Test Multi-Query Attention (n_kv_heads == 1)."""
        config = ModelConfig(d_model=256, n_layers=2, n_heads=4, n_kv_heads=1, attention_type="gqa")
        attn = build_attention(config)
        x = torch.randn(2, 64, 256)
        out = _forward(attn, x)
        assert out.shape == (2, 64, 256)


class TestMLA:
    """Tests for Multi-Head Latent Attention."""

    def test_forward_shape(self):
        config = ModelConfig(
            d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
            attention_type="mla",
            mla_kv_lora_rank=64, mla_q_lora_rank=128, mla_rope_head_dim=32,
        )
        attn = build_attention(config)
        x = torch.randn(2, 64, 256)
        out = _forward(attn, x)
        assert out.shape == (2, 64, 256)

    def test_backward(self):
        config = ModelConfig(
            d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
            attention_type="mla",
            mla_kv_lora_rank=64, mla_q_lora_rank=128, mla_rope_head_dim=32,
        )
        attn = build_attention(config)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = _forward(attn, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


@pytest.mark.parametrize("attention_type", ["mha", "gqa", "mla"])
def test_no_bias(attention_type):
    """Verify no bias in Q/K/V/O projections."""
    config = ModelConfig(
        d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
        attention_type=attention_type, use_bias=False,
        mla_kv_lora_rank=64, mla_q_lora_rank=128, mla_rope_head_dim=32,
    )
    attn = build_attention(config)
    for name, param in attn.named_parameters():
        if "bias" in name and "proj" in name:
            pytest.fail(f"Found bias parameter: {name}")
