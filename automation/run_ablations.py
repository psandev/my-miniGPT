"""Batch ablation runner for MiniGPT.

Reads ablation configurations from ``configs/ablations.yaml``, expands
variant matrices, runs each variant via ``run_experiment.py``, performs
pairwise judge comparisons, and generates a summary report.

Usage::

    python automation/run_ablations.py --config configs/ablations.yaml \\
        --output-dir experiments/ablations

    python automation/run_ablations.py --config configs/ablations.yaml \\
        --parallel 2 --resume
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig, TrainingConfig, PRESETS

logger = logging.getLogger(__name__)


def load_ablation_config(config_path: str) -> dict[str, Any]:
    """Load ablation configuration from a YAML file.

    Expected format::

        groups:
          norm_ablation:
            base_preset: small
            vary:
              norm_type: [rmsnorm, layernorm, dyt]
            dataset: tinystories
            max_steps: 10000

          attention_ablation:
            base_preset: small
            vary:
              attention_type: [mha, gqa, mla]
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def expand_variants(group_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a group's ``vary`` dict into individual variant configs.

    Parameters
    ----------
    group_config:
        A single ablation group with ``base_preset`` and ``vary`` dict.

    Returns
    -------
    list[dict[str, Any]]
        List of expanded variant configurations.
    """
    vary = group_config.get("vary", {})
    base_preset = group_config.get("base_preset", "small")

    # Generate all combinations of varied parameters
    param_names = list(vary.keys())
    param_values = [vary[k] if isinstance(vary[k], list) else [vary[k]] for k in param_names]

    variants = []
    for combo in itertools.product(*param_values):
        overrides = dict(zip(param_names, combo))
        variant_name = "_".join(f"{k}={v}" for k, v in overrides.items())

        variant = {
            "name": variant_name,
            "base_preset": base_preset,
            "overrides": overrides,
            "dataset": group_config.get("dataset", "smollm"),
            "max_steps": group_config.get("max_steps"),
            "tokenizer": group_config.get("tokenizer", "meta-llama/Llama-3.2-1B"),
        }
        variants.append(variant)

    return variants


def run_single_variant(
    variant: dict[str, Any],
    output_dir: str,
    use_wandb: bool = True,
) -> dict[str, Any]:
    """Run a single ablation variant experiment.

    Parameters
    ----------
    variant:
        Variant configuration dict.
    output_dir:
        Output directory for this variant.
    use_wandb:
        Whether to log to WandB.

    Returns
    -------
    dict[str, Any]
        Experiment summary for this variant.
    """
    from automation.run_experiment import run_experiment

    # Build model config from preset + overrides
    base_preset = variant["base_preset"]
    if base_preset in PRESETS:
        model_config = copy.deepcopy(PRESETS[base_preset])
    else:
        model_config = ModelConfig()

    for key, value in variant["overrides"].items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)

    training_config = TrainingConfig()
    if variant.get("max_steps"):
        training_config.max_steps = variant["max_steps"]

    tags = [f"ablation:{variant['name']}"]

    return run_experiment(
        model_config=model_config,
        training_config=training_config,
        dataset_name=variant.get("dataset", "smollm"),
        tokenizer_name=variant.get("tokenizer", "meta-llama/Llama-3.2-1B"),
        output_dir=output_dir,
        use_wandb=use_wandb,
        tags=tags,
    )


def run_ablation_group(
    group_name: str,
    group_config: dict[str, Any],
    output_dir: str,
    parallel: int = 1,
    resume: bool = False,
    use_wandb: bool = True,
    enable_judge: bool = False,
) -> dict[str, Any]:
    """Run all variants in an ablation group.

    Parameters
    ----------
    group_name:
        Name of the ablation group.
    group_config:
        Group configuration dict.
    output_dir:
        Root output directory for this group.
    parallel:
        Number of variants to run in parallel.
    resume:
        If ``True``, skip variants that already have a summary.json.
    use_wandb:
        Whether to log to WandB.
    enable_judge:
        Whether to run pairwise judge comparisons after all variants.

    Returns
    -------
    dict[str, Any]
        Group-level summary with all variant results.
    """
    variants = expand_variants(group_config)
    logger.info("Ablation group '%s': %d variants", group_name, len(variants))

    out = Path(output_dir) / group_name
    out.mkdir(parents=True, exist_ok=True)

    variant_results: dict[str, dict[str, Any]] = {}

    if parallel > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for variant in variants:
                variant_dir = str(out / variant["name"])

                # Skip if resume and already done
                if resume and (Path(variant_dir) / "summary.json").exists():
                    logger.info("Skipping completed variant: %s", variant["name"])
                    with open(Path(variant_dir) / "summary.json") as f:
                        variant_results[variant["name"]] = json.load(f)
                    continue

                future = executor.submit(
                    run_single_variant, variant, variant_dir, use_wandb,
                )
                futures[future] = variant["name"]

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    variant_results[name] = result
                    logger.info("Variant '%s' complete", name)
                except Exception as exc:
                    logger.error("Variant '%s' failed: %s", name, exc)
                    variant_results[name] = {"status": "failed", "error": str(exc)}
    else:
        # Sequential execution
        for variant in variants:
            variant_dir = str(out / variant["name"])

            if resume and (Path(variant_dir) / "summary.json").exists():
                logger.info("Skipping completed variant: %s", variant["name"])
                with open(Path(variant_dir) / "summary.json") as f:
                    variant_results[variant["name"]] = json.load(f)
                continue

            try:
                result = run_single_variant(variant, variant_dir, use_wandb)
                variant_results[variant["name"]] = result
                logger.info("Variant '%s' complete", variant["name"])
            except Exception as exc:
                logger.error("Variant '%s' failed: %s", variant["name"], exc)
                variant_results[variant["name"]] = {"status": "failed", "error": str(exc)}

    # Pairwise judge comparisons
    if enable_judge and len(variant_results) > 1:
        logger.info("Running pairwise judge comparisons")
        try:
            _run_pairwise_comparisons(variant_results, out)
        except Exception as exc:
            logger.warning("Pairwise judge failed: %s", exc)

    # Generate comparison table
    comparison = _generate_comparison_table(group_name, variant_results)

    # Group summary
    group_summary = {
        "group_name": group_name,
        "num_variants": len(variants),
        "variant_results": {k: {"status": v.get("status", "?")} for k, v in variant_results.items()},
        "comparison_table": comparison,
    }

    with open(out / "group_summary.json", "w") as f:
        json.dump(group_summary, f, indent=2, default=str)

    # Log to WandB
    if use_wandb:
        try:
            import wandb

            if wandb.run is None:
                wandb.init(project="minigpt-ablations")
            wandb.log({f"ablation/{group_name}/num_variants": len(variants)})
        except ImportError:
            pass

    return group_summary


def _run_pairwise_comparisons(
    variant_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Run pairwise judge comparisons between all variant pairs."""
    from evaluation.vertex_judge import evaluate_pairwise, JudgeConfig

    names = list(variant_results.keys())
    config = JudgeConfig(num_samples=50)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            samples_a = variant_results[name_a].get("generated_samples", [])
            samples_b = variant_results[name_b].get("generated_samples", [])

            if not samples_a or not samples_b:
                continue

            prompts = [s["prompt"] for s in samples_a]
            responses_a = [s["generated"] for s in samples_a]
            responses_b = [s["generated"] for s in samples_b]

            result = evaluate_pairwise(
                prompts=prompts,
                responses_a=responses_a,
                responses_b=responses_b,
                config=config,
                model_a_name=name_a,
                model_b_name=name_b,
            )

            comp_file = output_dir / f"judge_{name_a}_vs_{name_b}.json"
            with open(comp_file, "w") as f:
                json.dump(result, f, indent=2, default=str)


def _generate_comparison_table(
    group_name: str,
    variant_results: dict[str, dict[str, Any]],
) -> str:
    """Generate a markdown comparison table from variant results."""
    lines = [f"## Ablation: {group_name}\n"]

    # Perplexity comparison
    has_ppl = any("perplexity" in v for v in variant_results.values())
    has_bm = any("benchmarks" in v for v in variant_results.values())

    if has_ppl:
        lines.extend(["### Perplexity\n", "| Variant | WikiText-2 PPL |", "|---|---|"])
        for name, res in sorted(variant_results.items()):
            ppl = res.get("perplexity", {}).get("wikitext2", {}).get("perplexity", "N/A")
            lines.append(f"| {name} | {ppl} |")
        lines.append("")

    if has_bm:
        lines.extend(["### Benchmarks\n", "| Variant | Avg Accuracy |", "|---|---|"])
        for name, res in sorted(variant_results.items()):
            avg = res.get("benchmarks", {}).get("average_acc", "N/A")
            lines.append(f"| {name} | {avg} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="MiniGPT Ablation Runner")
    parser.add_argument("--config", required=True, help="Path to ablations.yaml")
    parser.add_argument("--output-dir", default="experiments/ablations")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel variant runs")
    parser.add_argument("--resume", action="store_true", help="Skip completed variants")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--enable-judge", action="store_true")
    parser.add_argument("--groups", nargs="*", default=None, help="Run only these groups")
    args = parser.parse_args()

    config = load_ablation_config(args.config)
    groups = config.get("groups", {})

    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}

    for group_name, group_config in groups.items():
        logger.info("Starting ablation group: %s", group_name)
        run_ablation_group(
            group_name=group_name,
            group_config=group_config,
            output_dir=args.output_dir,
            parallel=args.parallel,
            resume=args.resume,
            use_wandb=not args.no_wandb,
            enable_judge=args.enable_judge,
        )


if __name__ == "__main__":
    main()
