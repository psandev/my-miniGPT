# MiniGPT

A modular, research-oriented, decoder-only Transformer framework for pretraining, fine-tuning, aligning, evaluating, quantizing, and deploying small GPT models (40M – 3B parameters) on a single 24–48 GB GPU.

Every architectural component is swappable via a single config field. The factory pattern is used throughout — change one string and the entire model rebuilds with the new variant, no code edits required.

```
Input IDs
  → Token Embedding  (+ Learned PosEnc if applicable)
  → [mHC: expand to N streams]
  → Transformer Block × L:
        Norm → Attention  (+ RoPE / ALiBi) → Residual
        Norm → FFN                          → Residual
  → [mHC: collapse streams]
  → Prediction Head  (STP or MTP)
  → Logits + Loss
```

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
   - [Central config.yaml](#central-configyaml)
   - [ModelConfig](#modelconfig)
   - [TrainingConfig](#trainingconfig)
   - [DeployConfig](#deployconfig)
   - [Presets](#presets)
4. [Architecture: Swappable Components](#architecture-swappable-components)
   - [Attention](#attention)
   - [Feed-Forward Network](#feed-forward-network)
   - [Normalization](#normalization)
   - [Positional Encoding](#positional-encoding)
   - [Residual Connection](#residual-connection)
   - [Prediction Head](#prediction-head)
5. [Training](#training)
   - [Pretraining](#pretraining)
   - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
   - [DPO Alignment](#dpo-alignment)
   - [Datasets](#datasets)
6. [Evaluation](#evaluation)
   - [Automated Benchmarks](#automated-benchmarks)
   - [Perplexity](#perplexity)
   - [Gemini LLM-as-Judge](#gemini-llm-as-judge)
7. [Quantization](#quantization)
   - [HuggingFace Export](#huggingface-export)
   - [GGUF (llama.cpp / Ollama)](#gguf-llamacpp--ollama)
   - [AWQ (vLLM / SGLang)](#awq-vllm--sglang)
   - [GPTQ](#gptq)
   - [TurboQuant KV Cache Compression](#turboquant-kv-cache-compression)
8. [Deployment](#deployment)
   - [vLLM](#vllm)
   - [SGLang](#sglang)
   - [llama.cpp / Ollama](#llamacpp--ollama)
9. [Automation](#automation)
   - [Experiment Runner](#experiment-runner)
   - [Ablation Runner](#ablation-runner)
   - [WandB Sweep](#wandb-sweep)
   - [VRAM Estimator](#vram-estimator)
10. [Makefile Reference](#makefile-reference)
11. [Testing](#testing)
12. [Gradio Demo UI](#gradio-demo-ui)
13. [Dependencies](#dependencies)

---

## Quickstart

### 1. Set up the environment

```bash
conda activate mini_gpt_312
```

### 2. Configure paths

Edit `config.yaml` in the project root — only the `paths` section needs to match your machine:

```yaml
paths:
  data_dir:        /home/you/minigpt/data
  checkpoints_dir: /home/you/minigpt/checkpoints
  results_dir:     /home/you/minigpt/results
  pretrain_checkpoint: /home/you/minigpt/checkpoints/pretrain/tiny/final/checkpoint.pt
  sft_checkpoint:      /home/you/minigpt/checkpoints/sft/final/checkpoint.pt
  hf_export_dir:   /home/you/minigpt/checkpoints/hf_export
  gguf_path:       /home/you/minigpt/checkpoints/gguf/model-Q4_K_M.gguf
  awq_dir:         /home/you/minigpt/checkpoints/awq
```

All scripts read every other parameter directly from `config.yaml` — no CLI arguments required.

### 3. Prepare data

```bash
python data/prepare.py          # downloads datasets listed in config.yaml → data section
python data/prepare.py --status # check what's prepared
python data/prepare.py --list   # list all available datasets
```

### 4. Estimate VRAM before training

```bash
make estimate CONFIG=small
```

### 5. Pretrain

```bash
python training/train.py        # uses config.yaml → model.preset and training section
make train                      # same via Makefile
make train CONFIG=medium        # override preset
```

### 6. Fine-tune

```bash
python training/sft.py          # requires paths.pretrain_checkpoint to be set
python training/dpo.py          # requires paths.sft_checkpoint to be set
```

### 7. Evaluate

```bash
python evaluation/benchmarks.py   # uses paths.hf_export_dir
python evaluation/perplexity.py   # uses paths.pretrain_checkpoint
make eval
make eval-perplexity
```

### 8. Export and serve

```bash
make export-hf
make export-awq
make serve-vllm                 # uses paths.awq_dir
```

### 9. Interactive demo

```bash
make demo
```

### Python API

```python
from configs.model.config import PRESETS
from miniGPT.model import MiniGPT
from miniGPT.generation import generate_from_ids
import torch

# Build any preset
config = PRESETS["tiny"]
model  = MiniGPT(config)

# Forward pass
x       = torch.randint(0, config.vocab_size, (2, 64))
targets = torch.randint(0, config.vocab_size, (2, 64))
out     = model(x, targets=targets)
# out = {"logits": (2, 64, 32000), "loss": scalar, "aux_loss": scalar | None}

# Parameter count breakdown
print(model.count_parameters())
# {"embedding": "16.38M", "attention": "5.24M", "ffn": "18.87M", ...}

# Autoregressive generation (no tokenizer needed for quick testing)
ids     = torch.randint(0, config.vocab_size, (1, 8))
out_ids = generate_from_ids(model, ids, max_new_tokens=64, temperature=0.8)
```

---

## Project Structure

```
miniGPT/
├── config.yaml                  Central configuration (paths · data · model · training · sft · dpo · eval · judge · deploy)
│
├── configs/
│   ├── model/config.py          ModelConfig · TrainingConfig · DeployConfig · PRESETS (Python dataclasses)
│   ├── ablations.yaml           Ablation experiment matrix (read by automation/run_ablations.py)
│   └── sweep_config.yaml        WandB hyperparameter sweep (W&B sweep format)
│
├── data/
│   ├── prepare.py               Download · tokenize · pack datasets (pretrain .bin · SFT/DPO .jsonl)
│   └── */meta.json              Per-dataset metadata (committed; .bin/.jsonl gitignored)
│
├── miniGPT/
│   ├── model.py                 MiniGPT  (full model assembly)
│   ├── generation.py            generate() · generate_from_ids()
│   └── modules/
│       ├── attention.py         GroupedQueryAttention · MLAAttention
│       ├── ffn.py               SwiGLUFFN · GELUFFN · ReLUFFN · MoEFFN
│       ├── norms.py             RMSNorm · LayerNorm · DynamicTanh
│       ├── pos_encoding.py      RotaryPositionEncoding · LearnedPositionEncoding · ALiBiPositionEncoding
│       ├── residual.py          StandardResidual · ManifoldHyperConnection
│       └── prediction.py        SingleTokenPrediction · MultiTokenPrediction
│
├── training/
│   ├── train.py                 Pretraining entry point
│   ├── sft.py                   Supervised fine-tuning
│   ├── dpo.py                   DPO alignment (via TRL)
│   ├── data.py                  Dataset loaders for 8 datasets
│   ├── toml_generator.py        ModelConfig → TorchTitan TOML
│   └── torchtitan_integration.py FSDP2 · torch.compile · activation checkpointing
│
├── evaluation/
│   ├── benchmarks.py            lm-evaluation-harness wrapper
│   ├── perplexity.py            WikiText-2 / C4 sliding-window perplexity
│   ├── vertex_judge.py          Vertex AI pointwise + pairwise LLM-as-Judge
│   ├── llm_comparator.py        Google PAIR LLM Comparator HTML reports
│   └── report.py                WandB comparison tables
│
├── quantization/
│   ├── export_hf.py             MiniGPT checkpoint → HuggingFace / Llama format
│   ├── export_gguf.py           HF → GGUF  (Q4_K_M / Q6_K / Q4_K_S / ...)
│   ├── export_awq.py            HF → AWQ 4-bit  (Marlin kernel)
│   ├── export_gptq.py           HF → GPTQ 4-bit  (fallback)
│   └── turboquant.py            TurboQuant runtime KV-cache compression
│
├── deployment/
│   ├── serve_vllm.py            vLLM · OpenAI-compatible API
│   ├── serve_sglang.py          SGLang · RadixAttention
│   ├── serve_llamacpp.py        llama.cpp server · Ollama model registration
│   └── model_card.py            HuggingFace Hub model card generator
│
├── automation/
│   ├── run_experiment.py        End-to-end 10-step pipeline
│   ├── run_ablations.py         Batch ablation runner
│   ├── sweep.py                 WandB Bayesian hyperparameter sweep
│   └── memory_estimator.py      VRAM estimator (params + optim + activations)
│
├── ui/gradio_app.py             Interactive web demo
├── tests/                       148 unit tests (144 exhaustive shape combos)
├── scripts/                     download_data · train_tokenizer · upload_to_hub
├── Makefile
├── Dockerfile.train
├── Dockerfile.serve
├── pyproject.toml
└── requirements.txt
```

---

## Configuration

### Central config.yaml

`config.yaml` at the project root is the single source of truth. Every script loads it via OmegaConf at startup — no CLI arguments required.

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")

# Each section maps directly to a stage
cfg.paths      # environment-specific directories (only section that changes per machine)
cfg.data       # tokenizer, datasets, max_tokens, val_split
cfg.model      # architecture (preset or individual fields)
cfg.training   # lr, batch_size, max_steps, scheduler, ...
cfg.sft        # SFT lr, dataset, max_steps, ...
cfg.dpo        # beta, lr, dataset, ...
cfg.eval       # benchmark tasks, fewshot counts, perplexity dataset
cfg.judge      # Gemini judge auth, model, metrics
cfg.deploy     # backend, host, port, quantization, TurboQuant
```

CLI arguments still work as overrides when provided — the config is the default, not a hard constraint.

**Windows / multi-machine setup:** only the `paths` section needs changing. Everything else is committed and shared.

### Python Dataclasses (programmatic use)

`configs/model/config.py` contains typed Python dataclasses for building models in code.
No YAML parsing at model-build time — everything is typed and IDE-friendly.

### ModelConfig

The central config object that controls the entire model architecture.

```python
from configs.model.config import ModelConfig

config = ModelConfig(
    # --- Core dimensions ---
    vocab_size     = 32_000,
    d_model        = 2048,
    n_layers       = 24,
    n_heads        = 16,
    n_kv_heads     = 4,        # None -> MHA, < n_heads -> GQA, 1 -> MQA
    max_seq_len    = 4096,
    dropout        = 0.0,

    # --- Swappable components (change any of these freely) ---
    attention_type  = "gqa",       # "mha" | "gqa" | "mla"
    pos_encoding    = "rope",      # "rope" | "learned" | "alibi" | "none"
    norm_type       = "rmsnorm",   # "rmsnorm" | "layernorm" | "dyt"
    ffn_type        = "swiglu",    # "swiglu" | "gelu" | "relu" | "moe"
    residual_type   = "standard",  # "standard" | "mhc"
    prediction_type = "stp",       # "stp" | "mtp"

    # --- Global ---
    use_bias = False,              # No bias anywhere (modern default)

    # --- MLA-specific ---
    mla_kv_lora_rank  = 512,
    mla_q_lora_rank   = 1536,
    mla_rope_head_dim = 64,

    # --- RoPE-specific ---
    rope_base    = 10_000.0,
    rope_scaling = None,           # Set > 1.0 for context extension

    # --- DyT-specific ---
    dyt_alpha_init = 0.5,
    norm_eps       = 1e-6,

    # --- FFN dimensions ---
    ffn_multiplier = 2.667,        # d_ff = round_up_256(d_model x multiplier)

    # --- MoE-specific ---
    moe_num_experts    = 8,
    moe_top_k          = 2,
    moe_aux_loss_weight = 0.01,

    # --- mHC-specific ---
    mhc_n_streams      = 4,
    mhc_sinkhorn_iters = 5,

    # --- MTP-specific ---
    mtp_n_heads      = 4,
    mtp_loss_weight  = 1.0,
)

# Computed properties (read-only)
config.d_ff       # int  — GPU-aligned FFN hidden dim (nearest multiple of 256)
config.head_dim   # int  — d_model // n_heads
```

**Full field reference:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vocab_size` | int | 32000 | Vocabulary size |
| `d_model` | int | 2048 | Model hidden dimension |
| `n_layers` | int | 24 | Number of transformer blocks |
| `n_heads` | int | 16 | Number of attention query heads |
| `n_kv_heads` | int\|None | 4 | KV heads: None=MHA, <n_heads=GQA, 1=MQA |
| `max_seq_len` | int | 4096 | Maximum sequence length |
| `dropout` | float | 0.0 | Dropout probability (0.0 for most LLMs) |
| `attention_type` | str | `"gqa"` | Attention variant |
| `pos_encoding` | str | `"rope"` | Positional encoding type |
| `norm_type` | str | `"rmsnorm"` | Normalization layer type |
| `ffn_type` | str | `"swiglu"` | Feed-forward network type |
| `residual_type` | str | `"standard"` | Residual connection type |
| `prediction_type` | str | `"stp"` | Prediction head type |
| `use_bias` | bool | False | Bias in linear layers |
| `mla_kv_lora_rank` | int | 512 | MLA KV compression rank |
| `mla_q_lora_rank` | int | 1536 | MLA Q compression rank |
| `mla_rope_head_dim` | int | 64 | MLA decoupled RoPE head dim |
| `rope_base` | float | 10000.0 | RoPE frequency base |
| `rope_scaling` | float\|None | None | RoPE position scaling factor |
| `dyt_alpha_init` | float | 0.5 | DyT initial alpha value |
| `norm_eps` | float | 1e-6 | Normalization epsilon |
| `ffn_multiplier` | float | 2.667 | FFN hidden dim = d_model x multiplier |
| `moe_num_experts` | int | 8 | Number of MoE expert networks |
| `moe_top_k` | int | 2 | Top-k experts selected per token |
| `moe_aux_loss_weight` | float | 0.01 | Load-balancing loss coefficient |
| `mhc_n_streams` | int | 4 | Number of mHC residual streams |
| `mhc_sinkhorn_iters` | int | 5 | Sinkhorn-Knopp iterations |
| `mtp_n_heads` | int | 4 | Number of tokens predicted by MTP |
| `mtp_loss_weight` | float | 1.0 | MTP auxiliary loss weight |

### TrainingConfig

Controls the training loop, optimizer, and scheduler.

```python
from configs.model.config import TrainingConfig

tcfg = TrainingConfig(
    # Optimizer (AdamW)
    lr           = 3e-4,
    min_lr       = 3e-5,
    beta1        = 0.9,
    beta2        = 0.95,
    weight_decay = 0.1,
    grad_clip    = 1.0,

    # Schedule
    warmup_steps = 2_000,
    max_steps    = 100_000,
    scheduler    = "cosine",   # "cosine" | "linear" | "constant"

    # Batching
    batch_size                  = 32,
    seq_len                     = 4096,
    gradient_accumulation_steps = 1,

    # Precision & compilation
    precision                = "bf16",  # "bf16" | "fp16" | "fp32"
    activation_checkpointing = True,
    compile                  = True,

    # Logging & checkpointing
    log_interval  = 10,
    eval_interval = 500,
    save_interval = 1_000,
    wandb_project = "minigpt",
    output_dir    = "checkpoints",

    # Data
    dataset     = "tinystories",
    tokenizer   = "HuggingFaceTB/cosmo2-tokenizer",
    num_workers = 4,
)
```

### DeployConfig

Controls the serving backend and runtime quantization.

```python
from configs.model.config import DeployConfig

dcfg = DeployConfig(
    backend                = "vllm",   # "vllm" | "sglang" | "llamacpp"
    host                   = "0.0.0.0",
    port                   = 8000,
    max_batch_size         = 32,
    gpu_memory_utilization = 0.90,

    # Weight quantization format
    quantization     = "awq",    # "awq" | "gptq" | "gguf" | None
    gguf_quant_type  = "Q4_K_M",

    # TurboQuant runtime KV-cache compression
    turboquant_enabled         = False,
    turboquant_bits            = 4,
    turboquant_residual_window = 128,
)
```

### Presets

Ten ready-to-use configs accessible via `PRESETS["name"]`:

| Preset | d_model | Layers | Heads | KV Heads | ~Params | Notes |
|--------|---------|--------|-------|----------|---------|-------|
| `tiny` | 512 | 8 | 8 | 2 | 40 M | Fast experiments, CPU-runnable |
| `small` | 1024 | 16 | 16 | 4 | 350 M | Ablation target |
| `medium` | 2048 | 24 | 16 | 4 | 1.3 B | Main research target |
| `large` | 3072 | 32 | 24 | 8 | 3 B | Near upper limit for 48 GB GPU |
| `ablation_mhc` | 1024 | 16 | 16 | 4 | 350 M | `residual_type="mhc"` |
| `ablation_mtp` | 1024 | 16 | 16 | 4 | 350 M | `prediction_type="mtp"` |
| `ablation_dyt` | 1024 | 16 | 16 | 4 | 350 M | `norm_type="dyt"` |
| `ablation_mla` | 1024 | 16 | 16 | 4 | 350 M | `attention_type="mla"` |
| `ablation_moe` | 1024 | 16 | 16 | 4 | 350 M | `ffn_type="moe"` |
| `kitchen_sink` | 2048 | 24 | 16 | 4 | 1.3 B | `mhc` + `mtp` combined |

```python
from configs.model.config import PRESETS
from miniGPT.model import MiniGPT

model = MiniGPT(PRESETS["medium"])
print(model.count_parameters())
```

---

## Architecture: Swappable Components

### Component Matrix

| Component | Config Field | Options | Default | Factory |
|-----------|-------------|---------|---------|---------|
| Attention | `attention_type` | `mha`, `gqa`, `mla` | `gqa` | `build_attention(config)` |
| FFN | `ffn_type` | `swiglu`, `gelu`, `relu`, `moe` | `swiglu` | `build_ffn(config)` |
| Normalization | `norm_type` | `rmsnorm`, `layernorm`, `dyt` | `rmsnorm` | `build_norm(config, dim)` |
| Positional Encoding | `pos_encoding` | `rope`, `learned`, `alibi`, `none` | `rope` | `build_pos_encoding(config)` |
| Residual Connection | `residual_type` | `standard`, `mhc` | `standard` | `build_residual(config)` |
| Prediction Head | `prediction_type` | `stp`, `mtp` | `stp` | `build_head(config, emb_weight)` |

### Attention

**`GroupedQueryAttention`** — unified MHA / GQA / MQA:

- `n_kv_heads == n_heads` → Multi-Head Attention
- `n_kv_heads < n_heads` → Grouped-Query Attention; KV heads repeated via `repeat_interleave`
- `n_kv_heads == 1` → Multi-Query Attention
- Uses `F.scaled_dot_product_attention` — automatically dispatches to Flash Attention 2 or Memory-Efficient Attention when available, no extra dependency
- No bias in Q/K/V/O projections (`use_bias=False` default)
- RoPE applied to Q and K before attention; ALiBi bias added to attention logits
- Full KV-cache support via `forward_with_cache()` for autoregressive decoding

**`MLAAttention`** — Multi-head Latent Attention (DeepSeek-V2, 2024):

- Low-rank KV compression: `x -> c_kv (kv_lora_rank) -> {K_nope, V}` per head
- Decoupled RoPE: separate rope key dimensions computed from the latent projection
- Q path: `x -> q_down (q_lora_rank) -> {Q_nope, Q_rope}`
- Reduces KV cache memory footprint from `O(n_heads x head_dim)` to `O(kv_lora_rank)` per token

```python
config = ModelConfig(
    attention_type    = "mla",
    mla_kv_lora_rank  = 512,
    mla_q_lora_rank   = 1536,
    mla_rope_head_dim = 64,
)
```

### Feed-Forward Network

All variants use `d_ff = round_up_256(d_model x ffn_multiplier)`.

**`SwiGLUFFN`** — `down( SiLU(gate(x)) * up(x) )` — three matrices
- `ffn_multiplier=2.667` (default) gives 2/3 x 4 expansion for parameter parity with two-matrix FFNs
- Reference: Shazeer 2020

**`GELUFFN`** — `down( GELU(up(x)) )` — two matrices, 4x expansion

**`ReLUFFN`** — `down( ReLU(up(x)) )` — two matrices, 4x expansion

**`MoEFFN`** — Mixture-of-Experts over `moe_num_experts` SwiGLU sub-networks:
- Top-`moe_top_k` routing with a learned gating network
- Load-balancing auxiliary loss: `moe_aux_loss_weight x n_experts x sum(f_i x P_i)`
- Auxiliary loss collected per layer and added to the total model loss automatically
- Uses a Python loop over experts (production would use Megablocks or Scattermoe for GPU-efficient sparse dispatch)

```python
config = ModelConfig(
    ffn_type            = "moe",
    moe_num_experts     = 8,
    moe_top_k           = 2,
    moe_aux_loss_weight = 0.01,
)
```

### Normalization

**`RMSNorm`** — `weight * x * rsqrt(mean(x^2) + eps)` — no bias, no mean centering
- Default in LLaMA, Mistral, Qwen
- Reference: Zhang & Sennrich 2019

**`LayerNorm`** — wraps `torch.nn.LayerNorm` with standard affine parameters
- Reference: Ba et al. 2016

**`DynamicTanh (DyT)`** — `gamma * tanh(alpha * x) + beta` — normalisation-free
- Learnable scalar `alpha` (init to `dyt_alpha_init`), per-channel `gamma` and `beta`
- Achieves performance comparable to LayerNorm with no normalization statistics
- Reference: Nguyen et al., CVPR 2025

```python
config = ModelConfig(norm_type="dyt", dyt_alpha_init=0.5)
```

### Positional Encoding

**`RotaryPositionEncoding (RoPE)`** — applied to Q and K inside attention
- Precomputed complex-exponential frequency table
- `rope_base=10000.0` default; increase for longer contexts (e.g. 500000 for LLaMA 3)
- Optional `rope_scaling` for context length extension by position interpolation
- `apply_rotary(q, k, positions)` method available for direct use
- Reference: Su et al. 2021

**`LearnedPositionEncoding`** — `nn.Embedding(max_seq_len, d_model)` added to token embeddings
- Classic absolute positional encoding from the original Transformer

**`ALiBiPositionEncoding`** — fixed linear attention biases with geometric slopes per head
- `get_bias(seq_len)` returns `(1, n_heads, seq_len, seq_len)` bias tensor
- Added to attention logits; no position vectors in the residual stream
- Generalizes to sequences longer than those seen in training
- Reference: Press et al., ICLR 2022

**`None`** — `pos_encoding="none"` disables all positional encoding

### Residual Connection

**`StandardResidual`** — classic pre-norm residual: `output = x + sublayer(norm(x))`
- Stateless — the sublayer callable and norm module are passed into `forward()` each call

**`ManifoldHyperConnection (mHC)`** — maintains `mhc_n_streams` parallel residual streams
- Mixing matrix parameterized as `sigmoid(W)`, projected onto the Birkhoff polytope (doubly stochastic) via Sinkhorn-Knopp iterations
- Learnable `expand_proj` (d_model -> n_streams x d_model) and `collapse_proj` (n_streams x d_model -> d_model)
- Forward per block: mix streams via doubly-stochastic matrix -> collapse to `(B, S, D)` for sublayer -> broadcast result back to all streams
- Hidden state is `(B, S, N, D)` between transformer blocks, `(B, S, D)` inside each sublayer
- Reference: Godfrey et al. 2024

```python
config = ModelConfig(
    residual_type      = "mhc",
    mhc_n_streams      = 4,
    mhc_sinkhorn_iters = 5,
)
```

### Prediction Head

**`SingleTokenPrediction (STP)`** — standard language model head
- `norm(hidden) -> linear(d_model, vocab_size)`
- Output projection **weight-tied** to the token embedding matrix
- Returns `{"logits": (B, S, V), "loss": scalar | None}`
- Loss: cross-entropy with `-100` masking for padding / instruction tokens

**`MultiTokenPrediction (MTP)`** — predicts multiple future tokens simultaneously
- Main head: identical to STP, weight-tied, predicts token `t+1`
- `mtp_n_heads - 1` auxiliary heads predicting tokens `t+2` through `t+n`:
  `linear(d_model, d_model) -> norm -> linear(d_model, vocab_size)`
- Auxiliary losses averaged and scaled by `mtp_loss_weight`, added to total loss
- At inference: only the main head is used; auxiliary heads are stripped during HF export
- Reference: Gloeckle et al. 2024 (Meta); also used in DeepSeek-V3

```python
config = ModelConfig(
    prediction_type = "mtp",
    mtp_n_heads     = 4,    # predict t+1, t+2, t+3, t+4
    mtp_loss_weight = 1.0,
)
```

---

## Training

### Pretraining

```bash
# Run with zero arguments — all params from config.yaml
python training/train.py

# Via Makefile
make train                    # uses config.yaml → model.preset
make train CONFIG=medium      # override preset

# CLI overrides (any subset; rest still come from config.yaml)
python training/train.py --preset medium --lr 2e-4 --max-steps 50000

# Multi-GPU via TorchTitan (FSDP2)
python training/train.py --use-torchtitan
```

**Training loop features:**

| Feature | Detail |
|---------|--------|
| Optimizer | AdamW with fused CUDA kernels |
| Schedule | Cosine decay with linear warmup; min_lr = 10% of peak |
| Gradient clipping | Applied before each optimizer step (`grad_clip=1.0`) |
| Mixed precision | bf16 by default (fp16 or fp32 supported) |
| Activation checkpointing | Recomputes activations during backward, reduces memory 2-4x |
| torch.compile | Kernel fusion via `torch.compile` |
| Gradient accumulation | Run N micro-batches before each optimizer step |
| WandB logging | Loss, lr, grad norm, perplexity every `log_interval` steps |
| Checkpoint resume | `--resume checkpoints/medium/step_5000.pt` |

**CLI arguments for `training/train.py`** (all optional — defaults from `config.yaml`):

```
--preset          Override model preset (tiny/small/medium/large)
--toml            TorchTitan TOML config path (alternative to preset)
--dataset         Override dataset key
--lr              Override learning rate
--batch-size      Override batch size per device
--max-steps       Override total training steps
--warmup-steps    Override warmup steps
--grad-clip       Override gradient clipping threshold
--weight-decay    Override AdamW weight decay
--gradient-accumulation-steps
--max-seq-len     Override sequence length
--no-wandb        Disable WandB logging (overrides config.yaml → training.use_wandb)
--no-compile      Disable torch.compile
--resume          Resume from checkpoint path
--use-torchtitan  Use TorchTitan backend (FSDP2)
```

**TorchTitan TOML recipes** (generated automatically via `make generate-tomls`):

```toml
[model]
name = "minigpt_medium"
d_model = 2048
n_layers = 24
n_heads = 16
n_kv_heads = 4
# ... all architecture fields

[optimizer]
lr = 0.0003
betas = [0.9, 0.95]
weight_decay = 0.1

[training]
warmup_steps = 2000
max_steps = 100000
batch_size = 32
seq_len = 4096
precision = "bf16"
activation_checkpointing = true
compile = true

[metrics]
log_interval = 10
wandb_project = "minigpt"

[checkpoint]
save_interval = 1000
output_dir = "checkpoints/medium"
```

### Supervised Fine-Tuning (SFT)

Set `paths.pretrain_checkpoint` in `config.yaml`, then:

```bash
python training/sft.py          # all params from config.yaml
make train-sft                  # same via Makefile

# Override specific params
python training/sft.py --checkpoint /path/to/checkpoint.pt --max-steps 3000
```

**Key SFT settings:**
- Learning rate `1e-5` (30x lower than pretraining)
- Loss computed only on response tokens; instruction tokens masked with `-100`
- Chat template applied automatically:
  ```
  <|system|>
  You are a helpful assistant.
  <|user|>
  {instruction}
  <|assistant|>
  {response}
  ```
- Supported datasets: `alpaca` (Alpaca 52K), `oasst` (OpenAssistant OASST1), `oasst2`

### DPO Alignment

Set `paths.sft_checkpoint` in `config.yaml`, then:

```bash
python training/dpo.py          # all params from config.yaml
make train-dpo                  # same via Makefile

# Override specific params
python training/dpo.py --checkpoint /path/to/sft.pt --beta 0.05
```

**Key DPO settings:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `beta` | 0.1 | KL divergence penalty — lower = more deviation from reference |
| `lr` | 5e-6 | Very conservative; preserves SFT capabilities |
| Reference model | Frozen copy | Loaded from same SFT checkpoint |
| Dataset | Anthropic HH-RLHF | Parsed from `Human:/Assistant:` format |
| `max_prompt_length` | `max_seq_len // 2` | Half the context for response tokens |

Uses TRL `DPOTrainer` with bf16 on CUDA.

### Datasets

| Key | Source | Kind | Large? |
|-----|--------|------|--------|
| `tinystories` | roneneldan/TinyStories | Pretrain | No |
| `openwebtext_10k` | Skylion007/openwebtext | Pretrain | No (cap with `--max-tokens`) |
| `smollm` | HuggingFaceTB/smollm-corpus | Pretrain | Yes (~220 GB) |
| `c4` | allenai/c4 | Pretrain | Yes (~300 GB) |
| `alpaca` | tatsu-lab/alpaca | SFT | No |
| `oasst` | OpenAssistant/oasst1 | SFT | No |
| `oasst2` | OpenAssistant/oasst2 | SFT | No |
| `hh_rlhf` | Anthropic/hh-rlhf | DPO | No |

**`data/prepare.py`** — full data pipeline:
- **Pretraining**: streams from HuggingFace → tokenizes → packs into `uint16 .bin` files (nanoGPT-compatible memory-mapped format)
- **SFT**: streams → applies chat template → tokenizes → masks instruction tokens → saves `.jsonl`
- **DPO**: streams → parses `Human:/Assistant:` format → saves `.jsonl` with `{prompt, chosen, rejected}`

```bash
python data/prepare.py                          # prepare datasets listed in config.yaml → data.datasets
python data/prepare.py --datasets tinystories alpaca   # prepare specific datasets
python data/prepare.py --all                    # all non-large datasets
python data/prepare.py --all --include-large    # everything (200+ GB)
python data/prepare.py --status                 # show what's prepared
python data/prepare.py --list                   # list all available datasets
make data                                       # same as first line via Makefile
```

Output lands in `paths.data_dir` with one subdirectory per dataset containing `train.bin`/`val.bin` (pretrain) or `train.jsonl`/`val.jsonl` (SFT/DPO) plus `meta.json`.

---

## Evaluation

### Automated Benchmarks

Set `paths.hf_export_dir` in `config.yaml`, then:

```bash
python evaluation/benchmarks.py    # uses config.yaml → eval section
make eval

# Override model or tasks
python evaluation/benchmarks.py --model-path /other/model --tasks hellaswag,arc_easy
python evaluation/benchmarks.py --use-vllm   # faster vLLM backend
```

**Benchmark suite:**

| Benchmark | Measures | Metric | Few-shot |
|-----------|----------|--------|----------|
| HellaSwag | Commonsense NLI | acc_norm | 0 |
| ARC-Easy | Elementary science QA | acc | 0 |
| ARC-Challenge | Hard science QA | acc_norm | 0 |
| PIQA | Physical intuition | acc | 0 |
| WinoGrande | Pronoun resolution | acc | 0 |
| MMLU | 57-subject knowledge | acc | 5 |

Results are logged to WandB project `minigpt-eval` with experiment tags for comparison across runs.

### Perplexity

```bash
python evaluation/perplexity.py    # uses config.yaml → eval.perplexity_dataset + paths.pretrain_checkpoint
make eval-perplexity

# Override
python evaluation/perplexity.py --checkpoint /path/to/ckpt.pt --dataset c4
```

Implements **sliding-window perplexity** to handle sequences longer than `max_seq_len`:
- Window stride = `max_seq_len // 2` by default
- Only non-overlapping token positions contribute to the reported loss
- Supports WikiText-2 (validation) and C4 (streaming validation)

### Gemini LLM-as-Judge

Uses the `google-genai` SDK. Configure `config.yaml → judge`:

```yaml
judge:
  auth_mode: api_key        # api_key | vertex_ai
  api_key: ""               # or set GEMINI_API_KEY env var
  judge_model: gemini-2.0-flash
  num_samples: 100
  position_flip: true
  temperature: 0.0
```

**API key setup (easiest — free tier available):**
1. Go to https://aistudio.google.com/apikey
2. Create a key
3. `export GEMINI_API_KEY="your-key"`

**Vertex AI setup** (requires GCP project with Gemini in Model Garden):
1. `gcloud auth application-default login`
2. Set `auth_mode: vertex_ai` and `project_id` in `config.yaml`

**Pointwise evaluation** — each output scored independently on a 1–5 scale per metric:

```bash
make judge-pointwise MODEL=checkpoints/medium

python evaluation/vertex_judge.py \
    --mode pointwise \
    --input samples.jsonl \
    --output results/pointwise.json
```

**Pairwise A/B evaluation** — compares two variants with position-bias mitigation:

```bash
make judge MODEL_A=checkpoints/medium-sft MODEL_B=checkpoints/medium-dpo

python evaluation/vertex_judge.py \
    --mode pairwise \
    --input comparisons.jsonl \
    --output results/pairwise.json
```

Position-bias flipping: half of all calls present B before A. Win rates are debiased and reported with Wilson score 95% confidence intervals.

**Custom TinyStories rubric** — grammar, creativity, consistency, plot coherence (each 1–5):

```bash
python evaluation/vertex_judge.py \
    --mode tinystories \
    --input stories.jsonl
```

---

## Quantization

### HuggingFace Export

The required first step before any deployment — converts MiniGPT's checkpoint to standard Llama-compatible HuggingFace format:

```bash
make export-hf    # uses paths.pretrain_checkpoint → paths.hf_export_dir from config.yaml

# Override
python quantization/export_hf.py \
    --checkpoint /path/to/checkpoint.pt \
    --output-dir /path/to/hf_export \
    --dtype bfloat16
```

**What the export does:**

1. Strips MTP auxiliary prediction heads — only the main weight-tied head is kept
2. Collapses mHC multi-stream residual weights into standard single-stream parameters
3. Removes MoE auxiliary loss components
4. Remaps weight names to Llama HuggingFace format:
   - `tok_emb` -> `model.embed_tokens`
   - `attn.q_proj` -> `model.layers.N.self_attn.q_proj`
   - `ffn.gate_proj` -> `model.layers.N.mlp.gate_proj`
   - `attn_norm` -> `model.layers.N.input_layernorm`
5. Saves as `safetensors` with standard `config.json` (`architecture: LlamaForCausalLM`)
6. Writes `generation_config.json` (temperature=0.7, top_p=0.9, max_new_tokens=512)
7. Copies the tokenizer files and writes `minigpt_config.json` (original config for reference)

The exported model is a drop-in replacement for any Llama-architecture model in vLLM, SGLang, llama.cpp, or the HuggingFace `transformers` library.

### GGUF (llama.cpp / Ollama)

```bash
make export-gguf MODEL=checkpoints/medium-hf

# Specific quant type
python quantization/export_gguf.py \
    --model-path checkpoints/medium-hf \
    --output checkpoints/medium-Q4_K_M.gguf \
    --quant-type Q4_K_M

# Export all three common types at once
python quantization/export_gguf.py \
    --model-path checkpoints/medium-hf \
    --output checkpoints/ \
    --all-quants
```

| Quant Type | Size vs F16 | Quality | Recommended For |
|-----------|------------|---------|-----------------|
| `Q4_K_M` | ~25% | Good | Default, best quality/size balance |
| `Q6_K` | ~37% | Excellent | High quality, moderate compression |
| `Q4_K_S` | ~23% | Moderate | Maximum compression |
| `Q5_K_M` | ~31% | Very good | Middle ground |
| `Q8_0` | ~50% | Near-lossless | Validation / quality checks |
| `F16` | 100% | Lossless | Reference / comparison |

Requires `llama.cpp` installed. Auto-detected from `LLAMA_CPP_PATH` environment variable, `~/llama.cpp`, `/opt/llama.cpp`, or PATH.

### AWQ (vLLM / SGLang)

```bash
make export-awq MODEL=checkpoints/medium-hf

python quantization/export_awq.py \
    --model-path checkpoints/medium-hf \
    --output-dir checkpoints/medium-awq \
    --calibration-samples 128 \
    --bits 4 \
    --group-size 128 \
    --calibration-dataset pileval
```

- 4-bit weight quantization with 128-sample calibration
- Marlin kernel compatibility enabled by default for maximum throughput on Ampere/Ada GPUs
- Post-export verification with a forward pass
- Best serving option for vLLM and SGLang

### GPTQ

```bash
make export-gptq MODEL=checkpoints/medium-hf

python quantization/export_gptq.py \
    --model-path checkpoints/medium-hf \
    --output-dir checkpoints/medium-gptq
```

Fallback option if AWQ has compatibility issues. Uses `auto-gptq` with 4-bit quantization.

### TurboQuant KV Cache Compression

TurboQuant is a **runtime KV cache** compression hook — it compresses the attention KV cache in memory during inference, independently of weight quantization. This reduces the memory footprint of long-context inference.

```python
from quantization.turboquant import TurboQuantHook, TurboQuantConfig

cfg = TurboQuantConfig(
    bits             = 4,    # Compression bits (3-8, default 4)
    residual_window  = 256,  # Recent N tokens kept in full FP16
    group_size       = 128,
    use_rotation     = True, # PolarQuant random orthogonal rotation
    lloyd_max_iters  = 20,   # Quantizer fitting iterations
)

hook = TurboQuantHook(cfg)
hook.apply_to_model(model)   # Registers hooks on all attention layers

# Estimate memory savings before deploying
savings = hook.memory_savings_estimate(
    seq_len=4096, head_dim=128, n_kv_heads=4, n_layers=24
)
print(f"KV cache memory savings: {savings['savings_ratio']:.1%}")
```

**How it works:**

| Stage | Action |
|-------|--------|
| **PolarQuant rotation** | Random orthogonal rotation (QR decomposition) redistributes KV values into a more uniform distribution, reducing per-channel variance and improving quantization quality |
| **Lloyd-Max quantizer** | Iteratively fits optimal non-uniform quantization bins to the empirical KV distribution, minimizing mean-squared quantization error |
| **Residual window** | The most recent `residual_window` tokens are always stored in full FP16; only older tokens are quantized |
| **Per-head quantizers** | Separate K and V quantizers per attention head, each fitted to its own distribution for maximum quality |

Enable via `DeployConfig(turboquant_enabled=True)` or `--enable-turboquant` on any serve script.

---

## Deployment

The full train-to-deploy pipeline:

```
Pretrain  -->  SFT  -->  DPO  -->  export-hf  -->  [export-awq | export-gguf]  -->  serve
```

All training-specific components are removed at HF export:
- MTP auxiliary prediction heads (only main head kept)
- mHC multi-stream parameters (collapsed to standard single-stream)
- MoE auxiliary loss routing

The deployed model is structurally identical to a standard Llama architecture.

### vLLM

```bash
python deployment/serve_vllm.py    # uses config.yaml → deploy + paths.awq_dir
make serve-vllm

# Override
python deployment/serve_vllm.py --model /path/to/awq --port 9000 --enable-turboquant
```

Launches vLLM's OpenAI-compatible REST API. Use with the standard `openai` Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="minigpt",
    messages=[{"role": "user", "content": "Explain transformers simply."}],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

Quantization options: `awq_marlin` (recommended), `awq`, `gptq`, `none`.

### SGLang

```bash
python deployment/serve_sglang.py    # uses config.yaml → deploy + paths.awq_dir
make serve-sglang
```

SGLang features **RadixAttention** prefix caching — repeated system prompts, few-shot examples, or shared RAG context are cached and reused across requests. This significantly improves throughput for chat and RAG workloads with long shared prefixes.

Same OpenAI-compatible API endpoint as vLLM.

### llama.cpp / Ollama

```bash
python deployment/serve_llamacpp.py    # uses config.yaml → deploy + paths.gguf_path
make serve-ollama

# Override backend
python deployment/serve_llamacpp.py --backend llamacpp --model /path/to/model.gguf

# After registration
ollama run minigpt "Tell me a story about a robot."
```

The Ollama integration auto-generates a Modelfile with the chat template:

```
FROM /path/to/model.gguf
SYSTEM "You are a helpful assistant."
TEMPLATE "<|system|>\n{{ .System }}\n<|user|>\n{{ .Prompt }}\n<|assistant|>\n"
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```

---

## Automation

### Experiment Runner

Runs the full 10-step pipeline for a single model configuration:

```bash
python automation/run_experiment.py \
    --preset small \
    --dataset smollm \
    --output-dir runs/small-baseline \
    --enable-judge \
    --tags baseline small
```

**Pipeline steps:**

| Step | Action | Output File |
|------|--------|-------------|
| 1 | Resolve config (preset or TOML override) | — |
| 2 | Estimate VRAM; warn if insufficient | stdout |
| 3 | Generate TorchTitan TOML recipe | `recipe.toml` |
| 4 | Train model | `checkpoints/` |
| 5 | Export to HuggingFace format | `*-hf/` |
| 6 | Run benchmarks (HellaSwag, ARC, PIQA, MMLU, ...) | `benchmarks.json` |
| 7 | Compute WikiText-2 perplexity | `perplexity.json` |
| 8 | Generate text samples (3 default prompts) | `samples.json` |
| 9 | Vertex AI pointwise judge (if `--enable-judge`) | `judge_pointwise.json` |
| 10 | Log all results to WandB | `summary.json` |

### Ablation Runner

```bash
python automation/run_ablations.py \
    --config configs/ablations.yaml \
    --output-dir runs/ablations \
    --parallel 1 \        # or N for concurrent runs on multi-GPU systems
    --resume              # skip already-completed runs
```

For each ablation group in `configs/ablations.yaml`, the runner:
1. Expands all variant combinations
2. Runs each variant via `run_experiment.py`
3. After all variants complete, runs pairwise Vertex AI judge between all pairs
4. Generates a comparison table (perplexity, benchmark scores, win rates)
5. Logs the ablation summary to WandB

Example `ablations.yaml` structure:
```yaml
groups:
  - name: norm_comparison
    base_preset: small
    variants:
      - {norm_type: rmsnorm}
      - {norm_type: layernorm}
      - {norm_type: dyt}

  - name: ffn_comparison
    base_preset: small
    variants:
      - {ffn_type: swiglu}
      - {ffn_type: gelu}
      - {ffn_type: moe, moe_num_experts: 8}
```

### WandB Sweep

After identifying the best architecture from ablations, optimize hyperparameters:

```bash
make sweep
python automation/sweep.py --config configs/sweep_config.yaml --preset small --count 20
```

Bayesian search over:
- `learning_rate`: log-uniform in [1e-4, 5e-3]
- `warmup_steps`: uniform in [500, 5000]
- `batch_size`: categorical [16, 32, 64]
- `weight_decay`: uniform in [0.05, 0.2]

Optimization metric: validation perplexity (minimize).

### VRAM Estimator

```bash
make estimate CONFIG=medium
make estimate CONFIG=large

python automation/memory_estimator.py --preset large --batch-size 16 --dtype bf16 --json
```

Estimates every memory component:

| Component | Calculation |
|-----------|------------|
| Parameters | `n_params x bytes_per_param` |
| Gradients | same as parameters |
| Optimizer states | AdamW = 8 bytes/param (first + second moments in fp32) |
| Activations | `batch x seq x d_model x n_layers x 2` (with checkpointing: / n_layers) |
| KV Cache | `2 x n_kv_heads x head_dim x n_layers x seq x batch x 2` |
| mHC overhead | activation memory multiplied by `mhc_n_streams` |

The estimator reports:
- Total VRAM in GB
- Whether the config fits on a 24 GB or 48 GB GPU
- Recommended batch size for each tier

---

## Makefile Reference

All targets run under the `mini_gpt_312` conda environment automatically.
Default parameter values are read from `config.yaml`; Makefile variables override them.

### Data

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make data` | Prepare datasets from config | `DATASETS="a b c"`, `MAX_TOKENS=5000000` |
| `make data-all` | Prepare all non-large datasets | — |
| `make data-status` | Show preparation status | — |
| `make data-list` | List available datasets | — |

### Training

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make train` | Pretrain | `CONFIG=tiny\|small\|medium\|large` |
| `make train-sft` | Supervised fine-tuning | `MODEL=<checkpoint>` |
| `make train-dpo` | DPO alignment | `MODEL=<checkpoint>` |
| `make train-ablations` | Run full ablation matrix | — |

### Evaluation

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make eval` | Run all benchmarks | `MODEL=<hf-dir>` |
| `make eval-perplexity` | WikiText-2 perplexity | `MODEL=<checkpoint>` |
| `make judge` | Pairwise Gemini judge | `MODEL_A=`, `MODEL_B=` |
| `make judge-pointwise` | Pointwise quality scores | `MODEL=<hf-dir>` |
| `make report` | Generate evaluation report | — |

### Quantization & Export

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make export-hf` | Export to HuggingFace format | `MODEL=<checkpoint>` |
| `make export-gguf` | Convert to GGUF | `MODEL=<hf-dir>`, `QUANT_TYPE=Q4_K_M` |
| `make export-awq` | Convert to AWQ 4-bit | `MODEL=<hf-dir>` |
| `make export-gptq` | Convert to GPTQ 4-bit | `MODEL=<hf-dir>` |
| `make export-all` | HF + GGUF + AWQ | `MODEL=<checkpoint>` |

### Deployment

| Target | Description | Key Variables |
|--------|-------------|---------------|
| `make serve-vllm` | Launch vLLM server | `MODEL=<awq-dir>`, `PORT=8000` |
| `make serve-sglang` | Launch SGLang server | `MODEL=<awq-dir>`, `PORT=8000` |
| `make serve-ollama` | Launch via Ollama | `MODEL=<gguf-file>` |
| `make demo` | Gradio UI (port 7860) | — |

### Development & Utilities

| Target | Description |
|--------|-------------|
| `make test` | Run all unit tests |
| `make test-shapes` | Exhaustive 144-combo shape compatibility test |
| `make test-fast` | All tests except the slow shape sweep |
| `make estimate` | Estimate VRAM (`CONFIG=medium`) |
| `make generate-tomls` | Write TorchTitan TOML recipes for all presets |
| `make upload` | Upload model to HuggingFace Hub (`REPO_ID=user/repo`) |
| `make sweep` | Launch WandB hyperparameter sweep |
| `make docker-train` | Build training Docker image |
| `make docker-serve` | Build serving Docker image |
| `make clean` | Remove `__pycache__` and pytest cache |
| `make help` | Show all targets with descriptions |

---

## Testing

```bash
# Run all 148 tests
make test

# Run only the exhaustive shape compatibility test (144 combinations, ~5 seconds)
make test-shapes

# Run individual component tests
conda run -n mini_gpt_312 pytest tests/test_attention.py -v
conda run -n mini_gpt_312 pytest tests/test_model.py -v
```

**Full test suite:**

| File | Tests | What It Covers |
|------|-------|---------------|
| `test_attention.py` | 8 | MHA/GQA/MLA forward, backward, no-bias check |
| `test_ffn.py` | 12 | SwiGLU/GELU/ReLU/MoE forward, backward, aux loss |
| `test_norms.py` | 12 | RMSNorm/LayerNorm/DyT forward, backward, shapes |
| `test_residual.py` | 8 | Standard/mHC forward, expand/collapse, backward |
| `test_pos_encoding.py` | 12 | RoPE/Learned/ALiBi/None build, forward, backward |
| `test_prediction.py` | 8 | STP/MTP forward, backward, weight tying |
| `test_model.py` | 6 | Full model forward, loss, count_parameters |
| `test_generation.py` | 6 | Autoregressive generation, KV cache, sampling |
| `test_config.py` | 10 | All 10 presets instantiate without error |
| `test_shapes.py` | 148 | 144 exhaustive combos + 4 pos encoding variants |

The `test_shapes.py` exhaustive test is the most critical — it verifies every valid combination of all six swappable components:

```python
@pytest.mark.parametrize("attention",  ["mha", "gqa", "mla"])
@pytest.mark.parametrize("norm",       ["rmsnorm", "layernorm", "dyt"])
@pytest.mark.parametrize("ffn",        ["swiglu", "gelu", "relu", "moe"])
@pytest.mark.parametrize("residual",   ["standard", "mhc"])
@pytest.mark.parametrize("prediction", ["stp", "mtp"])
def test_full_forward(attention, norm, ffn, residual, prediction):
    config = ModelConfig(
        d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
        attention_type=attention, norm_type=norm, ffn_type=ffn,
        residual_type=residual, prediction_type=prediction,
    )
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    t = torch.randint(0, config.vocab_size, (2, 64))
    out = model(x, targets=t)
    assert out["logits"].shape == (2, 64, config.vocab_size)
    assert out["loss"] is not None
    out["loss"].backward()   # verify gradients flow through every combination
```

---

## Gradio Demo UI

```bash
make demo MODEL=checkpoints/medium-hf
# Opens at http://localhost:7860
```

**Features:**
- Dropdown to select from available model checkpoints
- Text prompt input with multi-line support
- Generation control sliders: temperature, top-k, top-p, max new tokens
- **Side-by-side comparison mode**: select two variants, enter one prompt, see outputs in parallel
- Model info panel showing parameter count breakdown and config summary

---

## Dependencies

### Core (always required)

```
torch >= 2.4          PyTorch with CUDA
transformers >= 4.40  HuggingFace model compatibility and tokenizers
datasets >= 2.18      Streaming dataset loading
tokenizers >= 0.15    Fast Rust-based tokenization
omegaconf >= 2.3      Central config.yaml loading (used by all scripts)
wandb >= 0.16         Experiment tracking and sweep management
trl >= 0.8            DPO trainer
```

**Default tokenizer:** `HuggingFaceTB/cosmo2-tokenizer` — open-access, 49K vocab, no login required. Change via `config.yaml → data.tokenizer`.

### Evaluation (`pip install "minigpt[eval]"`)

```
lm-eval >= 0.4                  EleutherAI lm-evaluation-harness
google-cloud-aiplatform >= 1.50  Vertex AI Gen AI Evaluation Service
```

### Quantization (`pip install "minigpt[quant]"`)

```
autoawq >= 0.2         AWQ 4-bit weight quantization (recommended)
auto-gptq >= 0.7       GPTQ 4-bit weight quantization (fallback)
llama-cpp-python >= 0.2  GGUF conversion testing
```

### Serving (`pip install "minigpt[serve]"`)

```
vllm >= 0.4    High-throughput LLM inference server
sglang >= 0.1  SGLang with RadixAttention prefix caching
gradio >= 4.0  Interactive web demo UI
```

### Install everything

```bash
# All optional dependencies at once
pip install -e ".[all]"

# Or use the pre-configured conda environment (already set up):
conda activate mini_gpt_312
```

### External tools (not pip-installable)

- **llama.cpp**: required for GGUF export. Set `LLAMA_CPP_PATH` or install to `~/llama.cpp`. See https://github.com/ggerganov/llama.cpp
- **Ollama**: optional, for `make serve-ollama`. See https://ollama.com
- **TorchTitan**: optional multi-GPU training backend. See https://github.com/pytorch/torchtitan

### Hardware requirements

| Preset | Params | Minimum VRAM | Recommended |
|--------|--------|-------------|-------------|
| `tiny` | 40 M | 4 GB | Any modern GPU |
| `small` | 350 M | 8 GB | RTX 3080 / A4000 |
| `medium` | 1.3 B | 24 GB | RTX 3090 / A5000 |
| `large` | 3 B | 48 GB | RTX 6000 Ada / A6000 |

Always run `make estimate CONFIG=<preset>` before training to verify your GPU can fit the chosen configuration with your desired batch size and sequence length.

---

## Docker

```bash
# Build images
make docker-build-train
make docker-build-serve

# Train inside container
docker run --gpus all \
    -v $(pwd)/checkpoints:/checkpoints \
    minigpt-train \
    python training/train.py --preset small

# Serve inside container
docker run --gpus all \
    -p 8000:8000 \
    -v $(pwd)/checkpoints/medium-awq:/model \
    minigpt-serve
```

`Dockerfile.train` — base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, installs all training dependencies  
`Dockerfile.serve` — base: vLLM official image, installs AWQ support, configurable entrypoint for vLLM or SGLang
