"""Tests for model configuration presets and config validation.

Verifies all presets instantiate without error and produce valid configurations.
"""

from __future__ import annotations

import pytest

from configs.model.config import ModelConfig, TrainingConfig, DeployConfig, PRESETS


class TestPresets:

    def test_all_presets_instantiate(self):
        """Every preset in PRESETS should instantiate without error."""
        for name, config in PRESETS.items():
            assert isinstance(config, ModelConfig), f"Preset '{name}' is not a ModelConfig"
            assert config.d_model > 0, f"Preset '{name}' has invalid d_model"
            assert config.n_layers > 0, f"Preset '{name}' has invalid n_layers"
            assert config.n_heads > 0, f"Preset '{name}' has invalid n_heads"

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_properties(self, preset_name):
        """Each preset should have valid derived properties."""
        config = PRESETS[preset_name]
        assert config.d_ff > 0
        assert config.head_dim > 0
        assert config.d_model % config.n_heads == 0, f"d_model not divisible by n_heads in '{preset_name}'"

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_valid_types(self, preset_name):
        """Each preset should have valid component type strings."""
        config = PRESETS[preset_name]
        assert config.attention_type in ("mha", "gqa", "mla")
        assert config.norm_type in ("rmsnorm", "layernorm", "dyt")
        assert config.ffn_type in ("swiglu", "gelu", "relu", "moe")
        assert config.pos_encoding in ("rope", "learned", "alibi", "none")
        assert config.residual_type in ("standard", "mhc")
        assert config.prediction_type in ("stp", "mtp")


class TestModelConfig:

    def test_default_config(self):
        config = ModelConfig()
        assert config.vocab_size == 32000
        assert config.d_model == 2048

    def test_d_ff_alignment(self):
        config = ModelConfig(d_model=1024)
        assert config.d_ff % 256 == 0, "d_ff should be 256-aligned for GPU efficiency"

    def test_head_dim(self):
        config = ModelConfig(d_model=256, n_heads=4)
        assert config.head_dim == 64

    def test_custom_config(self):
        config = ModelConfig(
            d_model=512, n_layers=8, n_heads=8, n_kv_heads=2,
            attention_type="gqa", norm_type="rmsnorm",
        )
        assert config.d_model == 512
        assert config.n_layers == 8
        assert config.n_kv_heads == 2


class TestTrainingConfig:

    def test_default_training_config(self):
        config = TrainingConfig()
        assert config.lr > 0
        assert config.batch_size > 0
        assert config.max_steps > 0


class TestDeployConfig:

    def test_default_deploy_config(self):
        config = DeployConfig()
        assert isinstance(config, DeployConfig)
