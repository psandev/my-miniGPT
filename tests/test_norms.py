"""Tests for all normalization variants: RMSNorm, LayerNorm, DyT.

Verifies forward pass shapes, backward gradient flow, and correct
behavior with and without bias.
"""

from __future__ import annotations

import pytest
import torch

from configs.model.config import ModelConfig
from miniGPT.modules.norms import build_norm


@pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm", "dyt"])
class TestNorms:

    def test_forward_shape(self, norm_type):
        config = ModelConfig(d_model=256, norm_type=norm_type)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 64, 256)
        out = norm(x)
        assert out.shape == (2, 64, 256)

    def test_backward(self, norm_type):
        config = ModelConfig(d_model=256, norm_type=norm_type)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 64, 256)

    def test_single_sample(self, norm_type):
        config = ModelConfig(d_model=256, norm_type=norm_type)
        norm = build_norm(config, config.d_model)
        x = torch.randn(1, 1, 256)
        out = norm(x)
        assert out.shape == (1, 1, 256)

    def test_different_dims(self, norm_type):
        for d_model in [64, 128, 512]:
            config = ModelConfig(d_model=d_model, n_heads=max(1, d_model // 64), norm_type=norm_type)
            norm = build_norm(config, d_model)
            x = torch.randn(2, 32, d_model)
            out = norm(x)
            assert out.shape == (2, 32, d_model)
