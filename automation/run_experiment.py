"""End-to-end experiment pipeline for MiniGPT.

Orchestrates the full lifecycle: config parsing, VRAM estimation, TOML generation,
training, benchmark evaluation, perplexity computation, sample generation,
optional Vertex AI judge evaluation, WandB logging, and summary JSON output.

Usage::

    python automation/run_experiment.py --preset small --dataset smollm \\
        --output-dir experiments/small_baseline

    python automation/run_experiment.py --preset small --enable-judge \\
        --output-dir experiments/small_judge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig, TrainingConfig, PRESETS

logger = logging.getLogger(__name__)


def run_experiment(
    preset: str | None = None,
    toml_path: str | None = None,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    dataset_name: str = "smollm",
    tokenizer_name: str = "meta-llama/Llama-3.2-1B",
    output_dir: str = "experiments/default",
    enable_benchmarks: bool = True,
    enable_perplexity: bool = True,
    enable_generation: bool = True,
    enable_judge: bool = False,
    judge_config_path: str | None = None,
    use_wandb: bool = True,
    tags: list[str] | None = None,
    generation_prompts: list[str] | None = None,
) -> dict[str, Any]:
    """Run a complete experiment pipeline.

    Parameters
    ----------
    preset:
        Model preset name from PRESETS.
    toml_path:
        Path to a TorchTitan TOML config (overrides preset).
    model_config:
        Explicit ModelConfig (overrides preset).
    training_config:
        Explicit TrainingConfig.
    dataset_name:
        Training dataset name.
    tokenizer_name:
        Tokenizer identifier.
    output_dir:
        Root directory for all experiment outputs.
    enable_benchmarks:
        Whether to run lm-evaluation-harness after training.
    enable_perplexity:
        Whether to compute perplexity on validation sets.
    enable_generation:
        Whether to generate text samples.
    enable_judge:
        Whether to run Vertex AI pointwise evaluation.
    judge_config_path:
        Path to judge config YAML.
    use_wandb:
        Whether to log to WandB.
    tags:
        WandB tags for the experiment.
    generation_prompts:
        Prompts for text generation samples.

    Returns
    -------
    dict[str, Any]
        Complete experiment summary.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"status": "running", "start_time": time.time()}

    # Step 1: Resolve config
    logger.info("Step 1: Resolving configuration")
    if model_config is None:
        if preset and preset in PRESETS:
            model_config = PRESETS[preset]
        else:
            model_config = PRESETS.get("tiny", ModelConfig())

    if training_config is None:
        training_config = TrainingConfig()

    summary["config"] = {
        "preset": preset,
        "model": model_config.__dict__,
        "training": training_config.__dict__,
        "dataset": dataset_name,
    }

    # Step 2: Estimate VRAM
    logger.info("Step 2: Estimating VRAM requirements")
    from automation.memory_estimator import estimate_memory

    mem_estimate = estimate_memory(model_config, training_config)
    summary["memory_estimate"] = mem_estimate
    logger.info("Estimated VRAM: %.1f GB", mem_estimate["total_gb"])

    if not mem_estimate.get("fits_24gb", True):
        logger.warning("Model may not fit on 24GB GPU! Consider reducing batch size.")

    # Step 3: Generate TOML if needed
    logger.info("Step 3: Generating TorchTitan TOML recipe")
    from training.toml_generator import generate_toml

    if toml_path is None:
        toml_path = str(out / "recipe.toml")
        generate_toml(
            model_config, training_config,
            output_path=toml_path,
            dataset_name=dataset_name,
            checkpoint_dir=str(out / "checkpoints"),
        )

    summary["toml_path"] = toml_path

    # Step 4: Train
    logger.info("Step 4: Training model")
    from training.train import train

    checkpoint_dir = train(
        model_config=model_config,
        training_config=training_config,
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        output_dir=str(out / "checkpoints"),
        use_wandb=use_wandb,
    )
    summary["checkpoint_dir"] = str(checkpoint_dir)

    checkpoint_path = str(checkpoint_dir / "checkpoint.pt")

    # Step 5: Export to HF format for benchmarks
    logger.info("Step 5: Exporting to HuggingFace format")
    hf_dir = out / "hf_export"
    try:
        from quantization.export_hf import export_to_hf

        export_to_hf(
            checkpoint_path=checkpoint_path,
            output_dir=str(hf_dir),
            tokenizer_name=tokenizer_name,
        )
        summary["hf_export_dir"] = str(hf_dir)
    except Exception as exc:
        logger.warning("HF export failed: %s", exc)
        summary["hf_export_error"] = str(exc)

    # Step 6: Run benchmarks
    if enable_benchmarks:
        logger.info("Step 6: Running benchmarks")
        try:
            from evaluation.benchmarks import run_benchmarks

            benchmark_results = run_benchmarks(
                model_path=str(hf_dir),
                output_path=str(out / "benchmarks.json"),
            )
            summary["benchmarks"] = benchmark_results
        except Exception as exc:
            logger.warning("Benchmark evaluation failed: %s", exc)
            summary["benchmark_error"] = str(exc)

    # Step 7: Compute perplexity
    if enable_perplexity:
        logger.info("Step 7: Computing perplexity")
        try:
            from evaluation.perplexity import evaluate_perplexity

            ppl_results = {}
            for dataset in ["wikitext2"]:
                result = evaluate_perplexity(
                    checkpoint_path=checkpoint_path,
                    dataset_name=dataset,
                    tokenizer_name=tokenizer_name,
                )
                ppl_results[dataset] = result

            summary["perplexity"] = ppl_results

            with open(out / "perplexity.json", "w") as f:
                json.dump(ppl_results, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Perplexity computation failed: %s", exc)
            summary["perplexity_error"] = str(exc)

    # Step 8: Generate samples
    if enable_generation:
        logger.info("Step 8: Generating text samples")
        try:
            import torch
            from miniGPT.model import MiniGPT
            from miniGPT.generation import generate
            from transformers import AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = MiniGPT(model_config).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

            prompts = generation_prompts or [
                "Once upon a time",
                "The meaning of life is",
                "In a world where AI",
            ]

            samples = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                output_ids = generate(model, input_ids, max_new_tokens=200, temperature=0.7, top_p=0.9)
                text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                samples.append({"prompt": prompt, "generated": text})

            summary["generated_samples"] = samples

            with open(out / "samples.json", "w") as f:
                json.dump(samples, f, indent=2)

            del model
        except Exception as exc:
            logger.warning("Text generation failed: %s", exc)
            summary["generation_error"] = str(exc)

    # Step 9: Optional Vertex AI judge
    if enable_judge:
        logger.info("Step 9: Running Vertex AI pointwise evaluation")
        try:
            from evaluation.vertex_judge import evaluate_pointwise, JudgeConfig

            judge_config = JudgeConfig()
            if judge_config_path:
                judge_config = JudgeConfig.from_yaml(judge_config_path)

            texts = [s["generated"] for s in summary.get("generated_samples", [])]
            prompts = [s["prompt"] for s in summary.get("generated_samples", [])]

            if texts:
                judge_results = evaluate_pointwise(
                    texts=texts, prompts=prompts, config=judge_config,
                )
                summary["judge_pointwise"] = judge_results

                with open(out / "judge_pointwise.json", "w") as f:
                    json.dump(judge_results, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Vertex AI judge failed: %s", exc)
            summary["judge_error"] = str(exc)

    # Step 10: Log to WandB
    if use_wandb:
        logger.info("Step 10: Logging to WandB")
        try:
            import wandb

            if wandb.run is None:
                wandb.init(project="minigpt", tags=tags or [])

            wandb.log({
                "experiment/preset": preset or "custom",
                "experiment/total_params": mem_estimate.get("total_params", 0),
            })

            if "benchmarks" in summary:
                for task, res in summary["benchmarks"].get("tasks", {}).items():
                    if res.get("acc") is not None:
                        wandb.log({f"experiment/benchmark/{task}": res["acc"]})

            if "perplexity" in summary:
                for ds, ppl in summary["perplexity"].items():
                    wandb.log({f"experiment/perplexity/{ds}": ppl.get("perplexity", 0)})
        except ImportError:
            logger.warning("wandb not installed")

    # Finalize summary
    summary["status"] = "complete"
    summary["end_time"] = time.time()
    summary["duration_seconds"] = summary["end_time"] - summary["start_time"]

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        "Experiment complete in %.1f seconds. Summary: %s",
        summary["duration_seconds"], out / "summary.json",
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="MiniGPT End-to-End Experiment Pipeline")
    parser.add_argument("--preset", type=str, default=None, help=f"Model preset: {list(PRESETS.keys())}")
    parser.add_argument("--toml", type=str, default=None, help="TorchTitan TOML path")
    parser.add_argument("--dataset", default="smollm")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="experiments/default")
    parser.add_argument("--no-benchmarks", action="store_true")
    parser.add_argument("--no-perplexity", action="store_true")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--enable-judge", action="store_true")
    parser.add_argument("--judge-config", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--tags", nargs="*", default=None)
    args = parser.parse_args()

    run_experiment(
        preset=args.preset,
        toml_path=args.toml,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        enable_benchmarks=not args.no_benchmarks,
        enable_perplexity=not args.no_perplexity,
        enable_generation=not args.no_generation,
        enable_judge=args.enable_judge,
        judge_config_path=args.judge_config,
        use_wandb=not args.no_wandb,
        tags=args.tags,
    )


if __name__ == "__main__":
    main()
