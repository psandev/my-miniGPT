"""Wrapper around EleutherAI lm-evaluation-harness for automated benchmarks.

Runs standard NLP benchmarks (HellaSwag, ARC, PIQA, WinoGrande, MMLU) against
a HuggingFace-format model checkpoint. Supports vLLM backend for speed.

Usage::

    python evaluation/benchmarks.py --model-path checkpoints/hf_export \\
        --tasks hellaswag,arc_easy,piqa --output results/benchmarks.json
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

# Standard benchmark suite
DEFAULT_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "mmlu",
]

# Number of few-shot examples per task
TASK_FEWSHOT: dict[str, int] = {
    "hellaswag": 0,
    "arc_easy": 0,
    "arc_challenge": 0,
    "piqa": 0,
    "winogrande": 0,
    "mmlu": 5,
}


def run_benchmarks(
    model_path: str,
    tasks: list[str] | None = None,
    batch_size: int | str = "auto",
    device: str = "cuda",
    use_vllm: bool = False,
    output_path: str | None = None,
    num_fewshot: dict[str, int] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run lm-evaluation-harness benchmarks on a model.

    Parameters
    ----------
    model_path:
        Path to a HuggingFace-format model directory.
    tasks:
        List of benchmark task names. Defaults to the standard suite.
    batch_size:
        Batch size for evaluation (``"auto"`` for automatic).
    device:
        Device string (``"cuda"`` or ``"cpu"``).
    use_vllm:
        If ``True``, use the vLLM backend for faster evaluation.
    output_path:
        If provided, save results JSON to this path.
    num_fewshot:
        Override the number of few-shot examples per task.
    limit:
        Limit the number of evaluation examples per task (for debugging).

    Returns
    -------
    dict[str, Any]
        Dictionary mapping task names to their results including accuracy
        and normalized accuracy.
    """
    import lm_eval  # type: ignore[import-untyped]

    if tasks is None:
        tasks = DEFAULT_TASKS

    if num_fewshot is None:
        num_fewshot = TASK_FEWSHOT

    logger.info("Running benchmarks: %s", tasks)
    logger.info("Model path: %s", model_path)
    logger.info("Backend: %s", "vLLM" if use_vllm else "HuggingFace")

    # Configure model arguments
    if use_vllm:
        model_args = f"pretrained={model_path},tensor_parallel_size=1,gpu_memory_utilization=0.8"
        model_type = "vllm"
    else:
        model_args = f"pretrained={model_path},dtype=bfloat16"
        model_type = "hf"

    # Run evaluation for each task group (different few-shot settings)
    all_results: dict[str, Any] = {}

    # Group tasks by fewshot count
    fewshot_groups: dict[int, list[str]] = {}
    for task in tasks:
        n = num_fewshot.get(task, 0)
        fewshot_groups.setdefault(n, []).append(task)

    for n_shot, task_group in fewshot_groups.items():
        task_str = ",".join(task_group)
        logger.info("Evaluating tasks [%s] with %d-shot", task_str, n_shot)

        eval_kwargs: dict[str, Any] = {
            "model": model_type,
            "model_args": model_args,
            "tasks": task_group,
            "num_fewshot": n_shot,
            "batch_size": batch_size,
            "device": device,
        }
        if limit is not None:
            eval_kwargs["limit"] = limit

        results = lm_eval.simple_evaluate(**eval_kwargs)

        # Extract results
        for task_name, task_results in results.get("results", {}).items():
            all_results[task_name] = {
                "acc": task_results.get("acc,none", task_results.get("acc", None)),
                "acc_norm": task_results.get("acc_norm,none", task_results.get("acc_norm", None)),
                "num_fewshot": n_shot,
            }

    # Build summary
    summary = {
        "model_path": model_path,
        "tasks": all_results,
        "average_acc": _compute_average(all_results, "acc"),
        "average_acc_norm": _compute_average(all_results, "acc_norm"),
    }

    logger.info("Benchmark results:")
    for task_name, res in all_results.items():
        acc = res.get("acc")
        acc_norm = res.get("acc_norm")
        logger.info("  %s: acc=%.4f, acc_norm=%s", task_name, acc or 0, f"{acc_norm:.4f}" if acc_norm else "N/A")

    logger.info("Average accuracy: %.4f", summary["average_acc"])

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return summary


def _compute_average(results: dict[str, Any], key: str) -> float:
    """Compute the average of a metric across tasks, ignoring None values."""
    values = [r[key] for r in results.values() if r.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def log_to_wandb(results: dict[str, Any], tags: list[str] | None = None) -> None:
    """Log benchmark results to Weights & Biases.

    Parameters
    ----------
    results:
        Benchmark results dict from ``run_benchmarks``.
    tags:
        Optional WandB tags for the run.
    """
    try:
        import wandb

        if wandb.run is None:
            wandb.init(project="minigpt-eval", tags=tags or [])

        log_dict: dict[str, float] = {}
        for task_name, task_res in results.get("tasks", {}).items():
            if task_res.get("acc") is not None:
                log_dict[f"benchmark/{task_name}/acc"] = task_res["acc"]
            if task_res.get("acc_norm") is not None:
                log_dict[f"benchmark/{task_name}/acc_norm"] = task_res["acc_norm"]

        log_dict["benchmark/average_acc"] = results.get("average_acc", 0.0)
        log_dict["benchmark/average_acc_norm"] = results.get("average_acc_norm", 0.0)

        wandb.log(log_dict)
        logger.info("Benchmark results logged to WandB")
    except ImportError:
        logger.warning("wandb not installed, skipping WandB logging")


def _load_cfg():
    """Load central config.yaml via OmegaConf (silent fallback if missing)."""
    try:
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        return OmegaConf.load(cfg_path)
    except Exception:
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load central config — evaluation parameters default from config.yaml
    cfg = _load_cfg()
    _e = cfg.eval   if cfg else None
    _p = cfg.paths  if cfg else None

    parser = argparse.ArgumentParser(description="MiniGPT Benchmark Evaluation")
    parser.add_argument("--model-path", default=None, help="Path to HF-format model")
    parser.add_argument("--tasks",      type=str, default=None,
                        help="Comma-separated task names")
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--use-vllm",   action="store_true")
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--wandb",      action="store_true")
    args = parser.parse_args()

    model_path = args.model_path or (str(_p.hf_export_dir) if _p else None)
    if not model_path:
        logger.error("No model path provided. Set paths.hf_export_dir in config.yaml or pass --model-path.")
        sys.exit(1)

    tasks = None
    if args.tasks:
        tasks = args.tasks.split(",")
    elif _e and _e.tasks:
        tasks = list(_e.tasks)

    output_path = args.output or (str(Path(_p.results_dir) / "benchmarks.json") if _p else None)

    results = run_benchmarks(
        model_path  = model_path,
        tasks       = tasks,
        batch_size  = args.batch_size or (str(_e.batch_size) if _e else "auto"),
        device      = args.device     or (str(_e.device)     if _e else "cuda"),
        use_vllm    = args.use_vllm   or (bool(_e.use_vllm)  if _e else False),
        output_path = output_path,
        limit       = args.limit      or (int(_e.limit) if _e and _e.limit else None),
    )

    use_wandb = args.wandb or (bool(_e.use_wandb) if _e else False)
    if use_wandb:
        log_to_wandb(results)


if __name__ == "__main__":
    main()
