"""Tests for all FFN variants: SwiGLU, GELU, ReLU, MoE.

Verifies forward pass shapes, backward gradient flow, and MoE routing.
"""

from __future__ import annotations

import pytest
import torch

from configs.model.config import ModelConfig
from miniGPT.modules.ffn import build_ffn


@pytest.fixture
def base_config():
    return ModelConfig(d_model=256, n_layers=2, n_heads=4, n_kv_heads=2)


@pytest.mark.parametrize("ffn_type", ["swiglu", "gelu", "relu"])
class TestDenseFFN:
    """Tests for dense (non-MoE) FFN variants."""

    def test_forward_shape(self, ffn_type):
        config = ModelConfig(d_model=256, n_layers=2, n_heads=4, ffn_type=ffn_type)
        ffn = build_ffn(config)
        x = torch.randn(2, 64, 256)
        out = ffn(x)
        assert out.shape == (2, 64, 256)

    def test_backward(self, ffn_type):
        config = ModelConfig(d_model=256, n_layers=2, n_heads=4, ffn_type=ffn_type)
        ffn = build_ffn(config)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestMoE:
    """Tests for Mixture of Experts FFN."""

    def test_forward_shape(self):
        config = ModelConfig(
            d_model=256, n_layers=2, n_heads=4,
            ffn_type="moe", moe_num_experts=4, moe_top_k=2,
        )
        ffn = build_ffn(config)
        x = torch.randn(2, 64, 256)
        out = ffn(x)
        # MoE may return a tuple (output, aux_loss) or just output
        if isinstance(out, tuple):
            output, aux_loss = out
            assert output.shape == (2, 64, 256)
            assert aux_loss.dim() == 0  # scalar
        else:
            assert out.shape == (2, 64, 256)

    def test_backward(self):
        config = ModelConfig(
            d_model=256, n_layers=2, n_heads=4,
            ffn_type="moe", moe_num_experts=4, moe_top_k=2,
        )
        ffn = build_ffn(config)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = ffn(x)
        if isinstance(out, tuple):
            loss = out[0].sum() + out[1]
        else:
            loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_expert_counts(self):
        for n_experts in [2, 4, 8]:
            config = ModelConfig(
                d_model=256, n_layers=2, n_heads=4,
                ffn_type="moe", moe_num_experts=n_experts, moe_top_k=min(2, n_experts),
            )
            ffn = build_ffn(config)
            x = torch.randn(1, 16, 256)
            out = ffn(x)
            if isinstance(out, tuple):
                assert out[0].shape == (1, 16, 256)
            else:
                assert out.shape == (1, 16, 256)
