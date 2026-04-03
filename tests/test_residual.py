"""Tests for residual connection variants: Standard and mHC.

Verifies forward pass shapes, backward gradient flow, and correct
handling of the multi-stream mHC residual connection.

The residual forward signature is: forward(x, sublayer_callable, norm)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from configs.model.config import ModelConfig
from miniGPT.modules.residual import build_residual
from miniGPT.modules.norms import build_norm


def _identity_sublayer(x: torch.Tensor):
    """Identity sublayer for testing — returns input unchanged."""
    return x


def _zero_sublayer(x: torch.Tensor):
    """Sublayer that returns zeros — useful for identity-preserving test."""
    return torch.zeros_like(x)


class TestStandardResidual:

    def test_forward(self):
        config = ModelConfig(d_model=256, residual_type="standard")
        residual = build_residual(config)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 64, 256)
        out = residual(x, _identity_sublayer, norm)
        assert out.shape == (2, 64, 256)

    def test_backward(self):
        config = ModelConfig(d_model=256, residual_type="standard")
        residual = build_residual(config)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = residual(x, _identity_sublayer, norm)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 64, 256)

    def test_residual_adds_input(self):
        """Standard residual: out = x + sublayer(norm(x)), so out != sublayer(norm(x))."""
        config = ModelConfig(d_model=256, residual_type="standard")
        residual = build_residual(config)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 16, 256)
        # With zero sublayer, out == x
        out = residual(x, _zero_sublayer, norm)
        assert torch.allclose(out, x, atol=1e-5)


class TestMHCResidual:
    """mHC residual operates on expanded (B, S, N, D) tensors.
    Use residual.expand() before and residual.collapse() after.
    """

    def test_expand_collapse(self):
        config = ModelConfig(d_model=256, residual_type="mhc", mhc_n_streams=4)
        residual = build_residual(config)
        x = torch.randn(2, 64, 256)
        expanded = residual.expand(x)
        assert expanded.shape == (2, 64, 4, 256)
        collapsed = residual.collapse(expanded)
        assert collapsed.shape == (2, 64, 256)

    def test_forward(self):
        config = ModelConfig(
            d_model=256, residual_type="mhc",
            mhc_n_streams=4, mhc_sinkhorn_iters=3,
        )
        residual = build_residual(config)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 64, 256)
        x_expanded = residual.expand(x)          # (B, S, N, D)
        out = residual(x_expanded, _identity_sublayer, norm)  # (B, S, N, D)
        assert out.shape == (2, 64, 4, 256)

    def test_forward_collapse(self):
        """Full round-trip: expand → mHC block → collapse."""
        config = ModelConfig(
            d_model=256, residual_type="mhc",
            mhc_n_streams=4, mhc_sinkhorn_iters=3,
        )
        residual = build_residual(config)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 64, 256)
        x_expanded = residual.expand(x)
        out_expanded = residual(x_expanded, _identity_sublayer, norm)
        out = residual.collapse(out_expanded)
        assert out.shape == (2, 64, 256)

    def test_backward(self):
        config = ModelConfig(
            d_model=256, residual_type="mhc",
            mhc_n_streams=4, mhc_sinkhorn_iters=3,
        )
        residual = build_residual(config)
        norm = build_norm(config, config.d_model)
        x = torch.randn(2, 64, 256, requires_grad=True)
        x_expanded = residual.expand(x)
        out = residual(x_expanded, _identity_sublayer, norm)
        loss = residual.collapse(out).sum()
        loss.backward()
        assert x.grad is not None

    def test_different_stream_counts(self):
        for n_streams in [2, 4, 8]:
            config = ModelConfig(
                d_model=256, residual_type="mhc",
                mhc_n_streams=n_streams,
            )
            residual = build_residual(config)
            norm = build_norm(config, config.d_model)
            x = torch.randn(2, 32, 256)
            x_expanded = residual.expand(x)
            out = residual(x_expanded, _identity_sublayer, norm)
            assert out.shape == (2, 32, n_streams, 256)


def test_standard_identity_preserving():
    """Standard residual with zero sublayer: out == x."""
    config = ModelConfig(d_model=256, residual_type="standard")
    residual = build_residual(config)
    norm = build_norm(config, config.d_model)
    x = torch.randn(2, 64, 256)
    out = residual(x, _zero_sublayer, norm)
    assert out.shape == (2, 64, 256)
    assert torch.allclose(out, x, atol=1e-5)
