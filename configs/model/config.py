"""
Model, training, and deployment configuration dataclasses for MiniGPT.

All architectural choices are controlled via ``ModelConfig`` fields and resolved
at runtime through factory functions (``build_attention``, ``build_ffn``, etc.).
Presets provide ready-made configurations for common model sizes and ablation
studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Complete architectural specification for a MiniGPT model.

    Every swappable component (attention, FFN, norm, positional encoding,
    residual connection, prediction head) is selected by a string field and
    instantiated through the corresponding ``build_*`` factory in
    ``miniGPT.modules``.

    Properties
    ----------
    d_ff : int
        Feed-forward hidden dimension, rounded up to the nearest multiple of
        256 for GPU-friendly alignment.
    head_dim : int
        Per-head dimension, derived from ``d_model // n_heads``.
    """

    # ---- Core dimensions ----
    vocab_size: int = 49_152  # matches HuggingFaceTB/cosmo2-tokenizer
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = 4  # None → MHA, < n_heads → GQA, 1 → MQA
    max_seq_len: int = 4096
    dropout: float = 0.0

    # ---- Swappable components ----
    attention_type: Literal["mha", "gqa", "mla"] = "gqa"
    pos_encoding: Literal["rope", "learned", "alibi", "none"] = "rope"
    norm_type: Literal["rmsnorm", "layernorm", "dyt"] = "rmsnorm"
    ffn_type: Literal["swiglu", "gelu", "relu", "moe"] = "swiglu"
    residual_type: Literal["standard", "mhc"] = "standard"
    prediction_type: Literal["stp", "mtp"] = "stp"

    # ---- Bias ----
    use_bias: bool = False

    # ---- MLA-specific ----
    mla_kv_lora_rank: int = 512
    mla_q_lora_rank: int = 1536
    mla_rope_head_dim: int = 64

    # ---- RoPE-specific ----
    rope_base: float = 10_000.0
    rope_scaling: Optional[float] = None

    # ---- DyT-specific ----
    dyt_alpha_init: float = 0.5
    norm_eps: float = 1e-6

    # ---- FFN dimensions ----
    ffn_multiplier: float = 2.667  # d_ff = d_model * multiplier (≈ 2/3 * 4 for SwiGLU parity)

    # ---- MoE-specific ----
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01

    # ---- mHC-specific ----
    mhc_n_streams: int = 4
    mhc_sinkhorn_iters: int = 5

    # ---- MTP-specific ----
    mtp_n_heads: int = 4
    mtp_loss_weight: float = 1.0

    @property
    def d_ff(self) -> int:
        """Feed-forward hidden dimension, rounded to nearest 256."""
        raw = int(self.d_model * self.ffn_multiplier)
        return ((raw + 255) // 256) * 256

    @property
    def head_dim(self) -> int:
        """Per-head dimension."""
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    """Training hyperparameters compatible with TorchTitan recipes.

    References
    ----------
    - Chinchilla scaling laws (Hoffmann et al., 2022)
    - GPT-3 training recipe (Brown et al., 2020)
    """

    # Optimiser
    lr: float = 3e-4
    min_lr: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    max_steps: int = 100_000
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # Batching
    batch_size: int = 32
    seq_len: int = 4096
    gradient_accumulation_steps: int = 1

    # Precision & memory
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    activation_checkpointing: bool = True
    compile: bool = True

    # Logging & checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    wandb_project: str = "minigpt"
    output_dir: str = "checkpoints"

    # Early stopping — stop if loss does not improve for this many log_intervals (0 = disabled)
    early_stopping_patience: int = 0

    # Data
    dataset: str = "HuggingFaceFW/fineweb-edu-dedup"
    tokenizer: str = "meta-llama/Llama-3.2-1B"
    num_workers: int = 4


@dataclass
class DeployConfig:
    """Deployment and serving configuration.

    Covers vLLM, SGLang, and llama.cpp backends as well as optional
    TurboQuant KV-cache compression.
    """

    backend: Literal["vllm", "sglang", "llamacpp"] = "vllm"
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    gpu_memory_utilization: float = 0.90

    # Quantisation
    quantization: Optional[Literal["awq", "gptq", "gguf"]] = None
    gguf_quant_type: str = "Q4_K_M"

    # TurboQuant KV-cache compression (experimental)
    turboquant_enabled: bool = False
    turboquant_bits: int = 4
    turboquant_residual_window: int = 128


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, ModelConfig] = {
    # ---- Size presets ----
    "tiny": ModelConfig(
        d_model=512, n_layers=8, n_heads=8, n_kv_heads=2,
    ),  # ~40 M
    "small": ModelConfig(
        d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
    ),  # ~350 M
    "medium": ModelConfig(
        d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
    ),  # ~1.3 B
    "large": ModelConfig(
        d_model=3072, n_layers=32, n_heads=24, n_kv_heads=8,
    ),  # ~3 B

    # ---- Ablation presets (small-sized for controlled comparison) ----
    "ablation_mhc": ModelConfig(
        d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
        residual_type="mhc",
    ),
    "ablation_mtp": ModelConfig(
        d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
        prediction_type="mtp",
    ),
    "ablation_dyt": ModelConfig(
        d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
        norm_type="dyt",
    ),
    "ablation_mla": ModelConfig(
        d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
        attention_type="mla",
    ),
    "ablation_moe": ModelConfig(
        d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
        ffn_type="moe",
    ),

    # ---- Kitchen-sink: modern defaults combined ----
    "kitchen_sink": ModelConfig(
        d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
        attention_type="gqa", norm_type="rmsnorm", ffn_type="swiglu",
        residual_type="mhc", prediction_type="mtp",
    ),
}
