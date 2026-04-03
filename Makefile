SHELL        := /bin/bash
CONDA        := conda run -n mini_gpt_312
PYTHON       := $(CONDA) python

# ── Defaults (override on the command line) ──────────────────────────────────
CONFIG       ?= tiny
DATASETS     ?= tinystories
MAX_TOKENS   ?= 5000000
MODEL        ?= $(OUT_DIR)/pretrain/$(CONFIG)/checkpoint.pt
MODEL_A      ?=
MODEL_B      ?=
TOKENIZER    ?= HuggingFaceTB/cosmo2-tokenizer
QUANT_TYPE   ?= Q4_K_M
HOST         ?= 0.0.0.0
PORT         ?= 8000
REPO_ID      ?=

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         := $(shell pwd)
OUT_DIR      := /home/access/peter/trd/miniGPT
DATA_DIR     := $(OUT_DIR)/data
CKPT_DIR     := $(OUT_DIR)/checkpoints
RESULTS_DIR  := $(OUT_DIR)/results

# ============================================================================
# Data
# ============================================================================

.PHONY: data
data: ## Download and tokenize datasets  (DATASETS="a b c"  MAX_TOKENS=5000000)
	$(PYTHON) data/prepare.py --datasets $(DATASETS) \
		--output-dir $(DATA_DIR) --tokenizer $(TOKENIZER) \
		--max-tokens $(MAX_TOKENS)

.PHONY: data-all
data-all: ## Prepare all non-large datasets
	$(PYTHON) data/prepare.py --all \
		--output-dir $(DATA_DIR) --tokenizer $(TOKENIZER)

.PHONY: data-status
data-status: ## Show which datasets have been prepared
	$(PYTHON) data/prepare.py --status --output-dir $(DATA_DIR)

.PHONY: data-list
data-list: ## List all available datasets
	$(PYTHON) data/prepare.py --list

# ============================================================================
# Training
# ============================================================================

.PHONY: train
train: ## Pretrain  (CONFIG=tiny|small|medium|large)
	$(PYTHON) training/train.py --preset $(CONFIG) \
		--data-dir $(DATA_DIR) --tokenizer $(TOKENIZER) \
		--output-dir $(CKPT_DIR)/pretrain/$(CONFIG)

.PHONY: train-sft
train-sft: ## Supervised fine-tuning  (MODEL=checkpoint path)
	$(PYTHON) training/sft.py --checkpoint $(MODEL) \
		--data-dir $(DATA_DIR) --tokenizer $(TOKENIZER) \
		--output-dir $(CKPT_DIR)/sft

.PHONY: train-dpo
train-dpo: ## DPO alignment  (MODEL=SFT checkpoint path)
	$(PYTHON) training/dpo.py --checkpoint $(MODEL) \
		--data-dir $(DATA_DIR) --tokenizer $(TOKENIZER) \
		--output-dir $(CKPT_DIR)/dpo

.PHONY: train-ablations
train-ablations: ## Run all ablations defined in configs/ablations.yaml
	$(PYTHON) automation/run_ablations.py --config configs/ablations.yaml \
		--output-dir $(OUT_DIR)/experiments/ablations

# ============================================================================
# Tests
# ============================================================================

.PHONY: test
test: ## Run full test suite
	$(CONDA) pytest tests/ -v --tb=short

.PHONY: test-shapes
test-shapes: ## Exhaustive shape-compatibility tests (148 combos)
	$(CONDA) pytest tests/test_shapes.py -v --tb=short

.PHONY: test-fast
test-fast: ## Run tests excluding slow shape sweep
	$(CONDA) pytest tests/ -v --tb=short --ignore=tests/test_shapes.py

# ============================================================================
# Evaluation
# ============================================================================

.PHONY: eval
eval: ## lm-evaluation-harness benchmarks  (MODEL=HF model path)
	$(PYTHON) evaluation/benchmarks.py --model-path $(MODEL) \
		--output $(RESULTS_DIR)/benchmarks.json

.PHONY: eval-perplexity
eval-perplexity: ## WikiText-2 perplexity  (MODEL=checkpoint path)
	$(PYTHON) evaluation/perplexity.py --checkpoint $(MODEL) \
		--dataset wikitext2 --output $(RESULTS_DIR)/perplexity.json

.PHONY: judge
judge: ## Pairwise Gemini judge  (MODEL_A=..., MODEL_B=...)
	$(PYTHON) evaluation/vertex_judge.py --mode pairwise \
		--input $(RESULTS_DIR)/responses.json \
		--output $(RESULTS_DIR)/judge_pairwise.json

.PHONY: judge-pointwise
judge-pointwise: ## Pointwise quality scoring  (MODEL=checkpoint)
	$(PYTHON) evaluation/vertex_judge.py --mode pointwise \
		--input $(RESULTS_DIR)/samples.json \
		--output $(RESULTS_DIR)/judge_pointwise.json

.PHONY: report
report: ## Generate evaluation report
	$(PYTHON) evaluation/report.py --results-dir $(RESULTS_DIR) \
		--output $(RESULTS_DIR)/report.md

# ============================================================================
# Export & Quantization
# ============================================================================

.PHONY: export-hf
export-hf: ## Export checkpoint to HuggingFace format  (MODEL=checkpoint)
	$(PYTHON) quantization/export_hf.py --checkpoint $(MODEL) \
		--output-dir $(CKPT_DIR)/hf_export --tokenizer $(TOKENIZER)

.PHONY: export-gguf
export-gguf: ## Convert to GGUF  (MODEL=HF model path, QUANT_TYPE=Q4_K_M)
	$(PYTHON) quantization/export_gguf.py --model-path $(MODEL) \
		--output $(CKPT_DIR)/gguf/model-$(QUANT_TYPE).gguf \
		--quant-type $(QUANT_TYPE)

.PHONY: export-awq
export-awq: ## Convert to AWQ 4-bit  (MODEL=HF model path)
	$(PYTHON) quantization/export_awq.py --model-path $(MODEL) \
		--output-dir $(CKPT_DIR)/awq

.PHONY: export-gptq
export-gptq: ## Convert to GPTQ 4-bit  (MODEL=HF model path)
	$(PYTHON) quantization/export_gptq.py --model-path $(MODEL) \
		--output-dir $(CKPT_DIR)/gptq

.PHONY: export-all
export-all: export-hf export-gguf export-awq ## Export to HF + GGUF + AWQ

# ============================================================================
# Deployment
# ============================================================================

.PHONY: serve-vllm
serve-vllm: ## Launch vLLM server  (MODEL=AWQ model path)
	$(PYTHON) deployment/serve_vllm.py --model $(MODEL) \
		--host $(HOST) --port $(PORT)

.PHONY: serve-sglang
serve-sglang: ## Launch SGLang server  (MODEL=AWQ model path)
	$(PYTHON) deployment/serve_sglang.py --model $(MODEL) \
		--host $(HOST) --port $(PORT)

.PHONY: serve-ollama
serve-ollama: ## Launch via Ollama/llama.cpp  (MODEL=GGUF path)
	$(PYTHON) deployment/serve_llamacpp.py --model $(MODEL) \
		--backend ollama --model-name minigpt

.PHONY: demo
demo: ## Launch Gradio chat UI
	$(PYTHON) ui/gradio_app.py --checkpoints-dir $(CKPT_DIR) --port 7860

# ============================================================================
# Docker
# ============================================================================

.PHONY: docker-train
docker-train: ## Build training Docker image
	docker build -t minigpt-train -f Dockerfile.train .

.PHONY: docker-serve
docker-serve: ## Build serving Docker image
	docker build -t minigpt-serve -f Dockerfile.serve .

# ============================================================================
# Utilities
# ============================================================================

.PHONY: estimate
estimate: ## Estimate VRAM for a preset  (CONFIG=tiny|small|medium|large)
	$(PYTHON) automation/memory_estimator.py --preset $(CONFIG)

.PHONY: sweep
sweep: ## Launch W&B hyperparameter sweep  (CONFIG=tiny)
	$(PYTHON) automation/sweep.py --config configs/sweep_config.yaml \
		--preset $(CONFIG) --count 20

.PHONY: upload
upload: ## Push model to HuggingFace Hub  (MODEL=HF path, REPO_ID=user/repo)
	$(PYTHON) scripts/upload_to_hub.py --model-path $(MODEL) --repo-id $(REPO_ID)

.PHONY: generate-tomls
generate-tomls: ## Regenerate TOML training recipes for all presets
	$(PYTHON) training/toml_generator.py --all --output-dir configs/training

.PHONY: clean
clean: ## Remove compiled Python files and pytest cache
	find $(ROOT) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find $(ROOT) -name "*.pyc" -delete 2>/dev/null; true
	rm -rf $(ROOT)/.pytest_cache $(ROOT)/.mypy_cache

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "  MiniGPT — available targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Key variables (current defaults):"
	@echo "    CONFIG=$(CONFIG)   DATASETS=$(DATASETS)   MAX_TOKENS=$(MAX_TOKENS)"
	@echo "    TOKENIZER=$(TOKENIZER)"
	@echo "    OUT_DIR=$(OUT_DIR)"
	@echo ""

.DEFAULT_GOAL := help
