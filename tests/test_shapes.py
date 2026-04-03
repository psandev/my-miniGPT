"""CRITICAL: Exhaustive parametrized shape compatibility test.

Tests every valid combination of attention, normalization, FFN, residual,
and prediction head types to verify that the full model forward pass
produces correct output shapes and that gradients flow correctly.

This test matrix covers:
- 3 attention types x 3 norm types x 4 FFN types x 2 residual types x 2 prediction types
  = 144 combinations
"""

from __future__ import annotations

import pytest
import torch

from configs.model.config import ModelConfig
from miniGPT.model import MiniGPT


@pytest.mark.parametrize("attention", ["mha", "gqa", "mla"])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm", "dyt"])
@pytest.mark.parametrize("ffn", ["swiglu", "gelu", "relu", "moe"])
@pytest.mark.parametrize("residual", ["standard", "mhc"])
@pytest.mark.parametrize("prediction", ["stp", "mtp"])
def test_full_forward(attention, norm, ffn, residual, prediction):
    """Test full model forward pass with every component combination.

    Verifies:
    1. Output logits have correct shape (batch, seq_len, vocab_size)
    2. Loss is computed when targets are provided
    3. Backward pass completes without error (gradients flow)
    """
    # Use small config for fast execution
    n_kv_heads = 4 if attention == "mha" else 2
    config = ModelConfig(
        d_model=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        max_seq_len=128,
        vocab_size=1000,
        dropout=0.0,
        attention_type=attention,
        norm_type=norm,
        ffn_type=ffn,
        residual_type=residual,
        prediction_type=prediction,
        use_bias=False,
        # MLA-specific (small values for fast test)
        mla_kv_lora_rank=64,
        mla_q_lora_rank=128,
        mla_rope_head_dim=32,
        # MoE-specific
        moe_num_experts=4,
        moe_top_k=2,
        moe_aux_loss_weight=0.01,
        # mHC-specific
        mhc_n_streams=4,
        mhc_sinkhorn_iters=3,
        # MTP-specific
        mtp_n_heads=3,
        mtp_loss_weight=1.0,
    )

    model = MiniGPT(config)
    model.train()

    # Create input and target tensors
    x = torch.randint(0, config.vocab_size, (2, 64))
    t = torch.randint(0, config.vocab_size, (2, 64))

    # Forward pass
    out = model(x, targets=t)

    # Verify output structure
    assert "logits" in out, f"Missing 'logits' key in output for {attention}/{norm}/{ffn}/{residual}/{prediction}"
    assert out["logits"].shape == (2, 64, config.vocab_size), (
        f"Wrong logits shape: {out['logits'].shape} for {attention}/{norm}/{ffn}/{residual}/{prediction}"
    )

    # Verify loss
    assert "loss" in out, f"Missing 'loss' key for {attention}/{norm}/{ffn}/{residual}/{prediction}"
    assert out["loss"] is not None, f"Loss is None for {attention}/{norm}/{ffn}/{residual}/{prediction}"
    assert out["loss"].dim() == 0, f"Loss is not scalar for {attention}/{norm}/{ffn}/{residual}/{prediction}"
    assert torch.isfinite(out["loss"]), f"Loss is not finite for {attention}/{norm}/{ffn}/{residual}/{prediction}"

    # Backward pass - verify gradients flow
    total_loss = out["loss"]
    aux_loss = out.get("aux_loss")
    if aux_loss is not None and torch.is_tensor(aux_loss) and aux_loss.item() > 0:
        total_loss = total_loss + aux_loss

    total_loss.backward()

    # Verify at least some parameters have gradients
    grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    total_count = sum(1 for p in model.parameters() if p.requires_grad)
    assert grad_count > 0, (
        f"No gradients computed for {attention}/{norm}/{ffn}/{residual}/{prediction} "
        f"(0/{total_count} params)"
    )


@pytest.mark.parametrize("pos_encoding", ["rope", "learned", "alibi", "none"])
def test_positional_encoding_variants(pos_encoding):
    """Test model with each positional encoding type."""
    config = ModelConfig(
        d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=128, vocab_size=1000,
        pos_encoding=pos_encoding,
    )
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    t = torch.randint(0, config.vocab_size, (2, 64))
    out = model(x, targets=t)
    assert out["logits"].shape == (2, 64, config.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()
