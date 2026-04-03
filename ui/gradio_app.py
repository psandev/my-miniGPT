"""Gradio web UI for MiniGPT text generation.

Interactive interface with:
- Dropdown to select model variant from available checkpoints
- Text input for prompt
- Sliders for generation parameters (temperature, top-k, top-p, max tokens)
- Side-by-side comparison mode for two variants
- Model info display (parameter count, config summary)

Usage::

    python ui/gradio_app.py --checkpoints-dir checkpoints --port 7860
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def discover_checkpoints(checkpoints_dir: str) -> dict[str, str]:
    """Discover available model checkpoints in a directory.

    Returns
    -------
    dict[str, str]
        Mapping of display name to checkpoint path.
    """
    models = {}
    base = Path(checkpoints_dir)
    if not base.exists():
        return models

    # Look for checkpoint.pt files
    for ckpt_file in sorted(base.glob("**/checkpoint.pt")):
        name = str(ckpt_file.parent.relative_to(base))
        models[name] = str(ckpt_file)

    # Look for HF-format models (with config.json)
    for config_file in sorted(base.glob("**/config.json")):
        parent = config_file.parent
        name = f"{parent.relative_to(base)} (HF)"
        if any(parent.glob("*.safetensors")) or any(parent.glob("*.bin")):
            models[name] = str(parent)

    return models


class ModelManager:
    """Manages loading and caching of models for the UI."""

    def __init__(self, checkpoints_dir: str) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.available_models = discover_checkpoints(checkpoints_dir)
        self._loaded_models: dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model_names(self) -> list[str]:
        return list(self.available_models.keys())

    def load_model(self, name: str) -> tuple[Any, Any, dict]:
        """Load a model by name, returning (model, tokenizer, config_dict)."""
        if name in self._loaded_models:
            return self._loaded_models[name]

        from configs.model.config import ModelConfig
        from miniGPT.model import MiniGPT
        from transformers import AutoTokenizer

        path = self.available_models.get(name)
        if not path:
            raise ValueError(f"Unknown model: {name}")

        if path.endswith(".pt"):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model_config = ModelConfig(**ckpt["model_config"])
            model = MiniGPT(model_config).to(self.device)
            model.load_state_dict(ckpt["model"])
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B", trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            config_dict = model_config.__dict__
            info = model.count_parameters()
        else:
            # HF format
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(self.device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            config_path = Path(path) / "minigpt_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
            else:
                config_dict = {}
            info = {"total": sum(p.numel() for p in model.parameters())}

        self._loaded_models[name] = (model, tokenizer, {**config_dict, "param_info": info})
        return self._loaded_models[name]

    def generate_text(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> str:
        """Generate text from a loaded model."""
        model, tokenizer, config = self.load_model(model_name)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        try:
            from miniGPT.generation import generate

            output_ids = generate(
                model, input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        except (ImportError, AttributeError):
            # Fallback to HF generate
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text

    def get_model_info(self, model_name: str) -> str:
        """Get formatted model info for display."""
        try:
            _, _, config = self.load_model(model_name)
            info_lines = []
            param_info = config.get("param_info", {})
            total = param_info.get("total", param_info.get("total_human", "?"))
            info_lines.append(f"Parameters: {total}")

            for key in ["d_model", "n_layers", "n_heads", "attention_type", "norm_type",
                        "ffn_type", "residual_type", "prediction_type", "max_seq_len"]:
                if key in config:
                    info_lines.append(f"{key}: {config[key]}")

            return "\n".join(info_lines)
        except Exception as exc:
            return f"Error loading model info: {exc}"


def create_app(checkpoints_dir: str = "checkpoints") -> Any:
    """Create the Gradio application.

    Parameters
    ----------
    checkpoints_dir:
        Directory containing model checkpoints.

    Returns
    -------
    gr.Blocks
        The Gradio app.
    """
    import gradio as gr

    manager = ModelManager(checkpoints_dir)
    model_names = manager.get_model_names()

    if not model_names:
        model_names = ["(no models found)"]

    with gr.Blocks(title="MiniGPT", theme=gr.themes.Soft()) as app:
        gr.Markdown("# MiniGPT Text Generation")
        gr.Markdown("Interactive demo for MiniGPT model variants.")

        with gr.Tabs():
            # Single model tab
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=model_names,
                            value=model_names[0] if model_names else None,
                            label="Model Variant",
                        )
                        model_info = gr.Textbox(label="Model Info", lines=8, interactive=False)
                        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                        top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
                        top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                        max_tokens = gr.Slider(16, 1024, value=256, step=16, label="Max Tokens")

                    with gr.Column(scale=2):
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3,
                        )
                        generate_btn = gr.Button("Generate", variant="primary")
                        output_text = gr.Textbox(label="Generated Text", lines=15)

                def on_generate(model_name, prompt, temp, tk, tp, mt):
                    if not prompt:
                        return "Please enter a prompt."
                    try:
                        return manager.generate_text(model_name, prompt, temp, tk, tp, mt)
                    except Exception as exc:
                        return f"Error: {exc}"

                def on_model_select(model_name):
                    return manager.get_model_info(model_name)

                generate_btn.click(
                    on_generate,
                    inputs=[model_dropdown, prompt_input, temperature, top_k, top_p, max_tokens],
                    outputs=output_text,
                )
                model_dropdown.change(on_model_select, inputs=model_dropdown, outputs=model_info)

            # Side-by-side comparison tab
            with gr.TabItem("Compare"):
                gr.Markdown("### Side-by-Side Comparison")
                with gr.Row():
                    model_a_dropdown = gr.Dropdown(choices=model_names, label="Model A")
                    model_b_dropdown = gr.Dropdown(choices=model_names, label="Model B")

                compare_prompt = gr.Textbox(label="Prompt", lines=3)

                with gr.Row():
                    compare_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                    compare_max = gr.Slider(16, 1024, value=256, step=16, label="Max Tokens")

                compare_btn = gr.Button("Compare", variant="primary")

                with gr.Row():
                    output_a = gr.Textbox(label="Model A Output", lines=12)
                    output_b = gr.Textbox(label="Model B Output", lines=12)

                def on_compare(model_a, model_b, prompt, temp, mt):
                    if not prompt:
                        return "Enter a prompt.", "Enter a prompt."
                    try:
                        text_a = manager.generate_text(model_a, prompt, temp, 50, 0.9, mt)
                    except Exception as exc:
                        text_a = f"Error: {exc}"
                    try:
                        text_b = manager.generate_text(model_b, prompt, temp, 50, 0.9, mt)
                    except Exception as exc:
                        text_b = f"Error: {exc}"
                    return text_a, text_b

                compare_btn.click(
                    on_compare,
                    inputs=[model_a_dropdown, model_b_dropdown, compare_prompt, compare_temp, compare_max],
                    outputs=[output_a, output_b],
                )

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="MiniGPT Gradio Web UI")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = create_app(args.checkpoints_dir)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
