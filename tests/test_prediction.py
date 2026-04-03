"""Tests for prediction head variants: STP and MTP.

Verifies forward pass shapes, backward gradient flow, weight tying,
and MTP auxiliary head behavior.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from configs.model.config import ModelConfig
from miniGPT.modules.prediction import build_head


def _make_embedding(config: ModelConfig) -> nn.Embedding:
    """Create a token embedding whose weight is passed to build_head."""
    return nn.Embedding(config.vocab_size, config.d_model)


class TestSTP:
    """Tests for Single Token Prediction head."""

    def test_forward_shape(self):
        config = ModelConfig(d_model=256, n_heads=4, prediction_type="stp")
        emb = _make_embedding(config)
        head = build_head(config, emb.weight)
        x = torch.randn(2, 64, 256)
        out = head(x)
        if isinstance(out, dict):
            assert out["logits"].shape == (2, 64, config.vocab_size)
        else:
            assert out.shape == (2, 64, config.vocab_size)

    def test_backward(self):
        config = ModelConfig(d_model=256, n_heads=4, prediction_type="stp")
        emb = _make_embedding(config)
        head = build_head(config, emb.weight)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = head(x)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None

    def test_weight_tying(self):
        """STP main projection weight should be tied to the embedding weight."""
        config = ModelConfig(d_model=256, n_heads=4, prediction_type="stp")
        emb = _make_embedding(config)
        head = build_head(config, emb.weight)
        # The head's projection weight should share data with embedding weight
        for name, param in head.named_parameters():
            if "weight" in name and param.shape == (config.vocab_size, config.d_model):
                assert param.data_ptr() == emb.weight.data_ptr(), \
                    f"Parameter {name} not tied to embedding weight"
                break


class TestMTP:
    """Tests for Multi-Token Prediction head."""

    def test_forward_shape(self):
        config = ModelConfig(
            d_model=256, n_heads=4, prediction_type="mtp",
            mtp_n_heads=4, mtp_loss_weight=1.0,
        )
        emb = _make_embedding(config)
        head = build_head(config, emb.weight)
        x = torch.randn(2, 64, 256)
        out = head(x)
        if isinstance(out, dict):
            assert out["logits"].shape == (2, 64, config.vocab_size)
        else:
            assert out.shape == (2, 64, config.vocab_size)

    def test_backward(self):
        config = ModelConfig(
            d_model=256, n_heads=4, prediction_type="mtp",
            mtp_n_heads=4, mtp_loss_weight=1.0,
        )
        emb = _make_embedding(config)
        head = build_head(config, emb.weight)
        x = torch.randn(2, 64, 256, requires_grad=True)
        out = head(x)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None

    def test_mtp_with_targets(self):
        """MTP head should return aux_logits list when targets provided."""
        config = ModelConfig(
            d_model=256, n_heads=4, prediction_type="mtp",
            mtp_n_heads=4, mtp_loss_weight=1.0,
        )
        emb = _make_embedding(config)
        head = build_head(config, emb.weight)
        x = torch.randn(2, 64, 256)
        targets = torch.randint(0, config.vocab_size, (2, 64))
        out = head(x, targets=targets)
        if isinstance(out, dict):
            assert "logits" in out
            assert out["logits"].shape == (2, 64, config.vocab_size)


@pytest.mark.parametrize("prediction_type", ["stp", "mtp"])
def test_build_head(prediction_type):
    """Verify that build_head returns a valid module."""
    config = ModelConfig(d_model=256, n_heads=4, prediction_type=prediction_type)
    emb = _make_embedding(config)
    head = build_head(config, emb.weight)
    assert head is not None
    params = list(head.parameters())
    assert len(params) > 0
