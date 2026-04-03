"""Generate HuggingFace model card for Hub upload.

Creates a standardized model card with architecture details, training info,
evaluation results, and usage examples.

Usage::

    python deployment/model_card.py --model-path checkpoints/hf_export \\
        --output checkpoints/hf_export/README.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def generate_model_card(
    model_path: str,
    model_name: str = "MiniGPT",
    author: str = "MiniGPT Team",
    description: str | None = None,
    eval_results: dict[str, Any] | None = None,
    training_info: dict[str, Any] | None = None,
    output_path: str | None = None,
) -> str:
    """Generate a HuggingFace model card.

    Parameters
    ----------
    model_path:
        Path to the HF-format model directory.
    model_name:
        Display name for the model.
    author:
        Model author or team name.
    description:
        Optional model description.
    eval_results:
        Optional evaluation results dict.
    training_info:
        Optional training details dict.
    output_path:
        If provided, write the card to this path.

    Returns
    -------
    str
        The model card content as a markdown string.
    """
    model_dir = Path(model_path)

    # Load model config
    config_path = model_dir / "config.json"
    minigpt_config_path = model_dir / "minigpt_config.json"
    config = {}
    minigpt_config = {}

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    if minigpt_config_path.exists():
        with open(minigpt_config_path) as f:
            minigpt_config = json.load(f)

    hidden_size = config.get("hidden_size", minigpt_config.get("d_model", "?"))
    n_layers = config.get("num_hidden_layers", minigpt_config.get("n_layers", "?"))
    n_heads = config.get("num_attention_heads", minigpt_config.get("n_heads", "?"))
    vocab_size = config.get("vocab_size", minigpt_config.get("vocab_size", "?"))
    max_seq = config.get("max_position_embeddings", minigpt_config.get("max_seq_len", "?"))

    if description is None:
        description = (
            f"{model_name} is a decoder-only Transformer language model built with the "
            f"MiniGPT framework, a modular research-oriented LLM toolkit."
        )

    # Build eval results table
    eval_table = ""
    if eval_results:
        eval_table = "## Evaluation Results\n\n| Benchmark | Score |\n|---|---|\n"
        for task, score in eval_results.items():
            if isinstance(score, dict):
                score_val = score.get("acc", score.get("score", "?"))
            else:
                score_val = score
            eval_table += f"| {task} | {score_val} |\n"
        eval_table += "\n"

    # Build training info section
    training_section = ""
    if training_info:
        training_section = "## Training Details\n\n"
        for key, value in training_info.items():
            training_section += f"- **{key}**: {value}\n"
        training_section += "\n"

    card = f"""---
library_name: transformers
tags:
  - minigpt
  - causal-lm
  - decoder-only
  - research
license: apache-2.0
---

# {model_name}

{description}

## Model Details

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer (Llama-compatible) |
| Hidden Size | {hidden_size} |
| Layers | {n_layers} |
| Attention Heads | {n_heads} |
| Vocabulary Size | {vocab_size} |
| Max Sequence Length | {max_seq} |

### MiniGPT Architecture

This model was built using the MiniGPT framework with the following component choices:

- **Attention**: {minigpt_config.get('attention_type', 'gqa')}
- **Normalization**: {minigpt_config.get('norm_type', 'rmsnorm')}
- **FFN**: {minigpt_config.get('ffn_type', 'swiglu')}
- **Positional Encoding**: {minigpt_config.get('pos_encoding', 'rope')}
- **Residual Connection**: {minigpt_config.get('residual_type', 'standard')}
- **Prediction Head**: {minigpt_config.get('prediction_type', 'stp')}

{eval_table}
{training_section}
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Quantized Versions

This model is also available in quantized formats:
- **AWQ 4-bit** (for vLLM/SGLang): Marlin kernel compatible
- **GGUF Q4_K_M** (for llama.cpp/Ollama): Best quality/size balance
- **GPTQ 4-bit** (fallback): For compatibility

## Framework

Built with [MiniGPT](https://github.com/minigpt) -- a modular, research-oriented
decoder-only Transformer framework for pretraining, fine-tuning, alignment,
evaluation, quantization, and deployment.

## License

Apache 2.0
"""

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(card)
        logger.info("Model card written to %s", output_path)

    return card


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate HuggingFace model card")
    parser.add_argument("--model-path", required=True, help="HF model directory")
    parser.add_argument("--model-name", default="MiniGPT")
    parser.add_argument("--author", default="MiniGPT Team")
    parser.add_argument("--output", default=None, help="Output path (default: model_path/README.md)")
    parser.add_argument("--eval-results", default=None, help="Eval results JSON file")
    args = parser.parse_args()

    eval_results = None
    if args.eval_results:
        with open(args.eval_results) as f:
            eval_results = json.load(f)

    output_path = args.output or str(Path(args.model_path) / "README.md")

    generate_model_card(
        model_path=args.model_path,
        model_name=args.model_name,
        author=args.author,
        eval_results=eval_results,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
