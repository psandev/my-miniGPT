"""Report generation for MiniGPT evaluation results.

Generates comparison tables, WandB reports, and markdown summaries
combining benchmark scores, perplexity values, and judge evaluations.

Usage::

    python evaluation/report.py --results-dir results/ --output results/report.md
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


def load_results(results_dir: str) -> dict[str, Any]:
    """Load all evaluation result files from a directory.

    Looks for files matching patterns like ``*benchmarks*.json``,
    ``*perplexity*.json``, ``*judge*.json``.

    Parameters
    ----------
    results_dir:
        Path to the directory containing result JSON files.

    Returns
    -------
    dict[str, Any]
        Aggregated results keyed by result type and model name.
    """
    results_path = Path(results_dir)
    aggregated: dict[str, Any] = {
        "benchmarks": {},
        "perplexity": {},
        "judge_pointwise": {},
        "judge_pairwise": {},
    }

    for json_file in sorted(results_path.glob("**/*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", json_file, exc)
            continue

        name = json_file.stem
        if "benchmark" in name:
            aggregated["benchmarks"][name] = data
        elif "perplexity" in name:
            aggregated["perplexity"][name] = data
        elif "judge" in name and "pairwise" in name:
            aggregated["judge_pairwise"][name] = data
        elif "judge" in name:
            aggregated["judge_pointwise"][name] = data

    return aggregated


def generate_comparison_table(
    results: dict[str, Any],
    models: list[str] | None = None,
) -> str:
    """Generate a markdown comparison table from aggregated results.

    Parameters
    ----------
    results:
        Aggregated results from ``load_results``.
    models:
        Optional list of model names to include. If ``None``, includes all.

    Returns
    -------
    str
        Markdown-formatted comparison table.
    """
    lines = ["## Model Comparison\n"]

    # Benchmark table
    benchmarks = results.get("benchmarks", {})
    if benchmarks:
        # Collect all task names
        all_tasks: set[str] = set()
        for bm in benchmarks.values():
            all_tasks.update(bm.get("tasks", {}).keys())

        tasks = sorted(all_tasks)
        header = "| Model | " + " | ".join(tasks) + " | Avg |"
        separator = "|" + "---|" * (len(tasks) + 2)
        lines.extend(["### Benchmark Accuracy\n", header, separator])

        for model_name, bm in sorted(benchmarks.items()):
            if models and model_name not in models:
                continue
            task_scores = []
            for task in tasks:
                acc = bm.get("tasks", {}).get(task, {}).get("acc")
                task_scores.append(f"{acc:.3f}" if acc is not None else "---")
            avg = bm.get("average_acc", 0)
            lines.append(f"| {model_name} | " + " | ".join(task_scores) + f" | {avg:.3f} |")
        lines.append("")

    # Perplexity table
    perplexity = results.get("perplexity", {})
    if perplexity:
        lines.extend([
            "### Perplexity\n",
            "| Model | Dataset | Perplexity | Loss |",
            "|---|---|---|---|",
        ])
        for model_name, ppl in sorted(perplexity.items()):
            if models and model_name not in models:
                continue
            dataset = ppl.get("dataset", "?")
            pp_val = ppl.get("perplexity", float("inf"))
            loss_val = ppl.get("loss", float("inf"))
            lines.append(f"| {model_name} | {dataset} | {pp_val:.2f} | {loss_val:.4f} |")
        lines.append("")

    # Judge pairwise results
    pairwise = results.get("judge_pairwise", {})
    if pairwise:
        lines.extend([
            "### Pairwise Judge Results\n",
            "| Comparison | Model A Win% | Model B Win% | Ties |",
            "|---|---|---|---|",
        ])
        for comp_name, comp in sorted(pairwise.items()):
            model_a = comp.get("model_a", "A")
            model_b = comp.get("model_b", "B")
            win_a = comp.get("win_rate_a", 0) * 100
            win_b = comp.get("win_rate_b", 0) * 100
            ties = comp.get("ties", 0)
            lines.append(f"| {model_a} vs {model_b} | {win_a:.1f}% | {win_b:.1f}% | {ties} |")
        lines.append("")

    return "\n".join(lines)


def generate_markdown_summary(
    results: dict[str, Any],
    title: str = "MiniGPT Evaluation Report",
) -> str:
    """Generate a full markdown evaluation report.

    Parameters
    ----------
    results:
        Aggregated results from ``load_results``.
    title:
        Report title.

    Returns
    -------
    str
        Complete markdown report.
    """
    lines = [f"# {title}\n"]

    # Summary statistics
    n_benchmarks = len(results.get("benchmarks", {}))
    n_perplexity = len(results.get("perplexity", {}))
    n_pairwise = len(results.get("judge_pairwise", {}))
    n_pointwise = len(results.get("judge_pointwise", {}))

    lines.extend([
        "## Summary\n",
        f"- **Benchmark evaluations**: {n_benchmarks}",
        f"- **Perplexity evaluations**: {n_perplexity}",
        f"- **Pairwise judge comparisons**: {n_pairwise}",
        f"- **Pointwise judge evaluations**: {n_pointwise}",
        "",
    ])

    # Comparison tables
    lines.append(generate_comparison_table(results))

    # Pointwise judge details
    pointwise = results.get("judge_pointwise", {})
    if pointwise:
        lines.extend(["### Pointwise Quality Scores\n"])
        for model_name, pw_results in sorted(pointwise.items()):
            lines.append(f"#### {model_name}\n")
            if isinstance(pw_results, list):
                # Aggregate scores across samples
                metric_scores: dict[str, list[float]] = {}
                for sample in pw_results:
                    for key, value in sample.items():
                        if isinstance(value, dict) and "score" in value and value["score"] is not None:
                            metric_scores.setdefault(key, []).append(value["score"])

                if metric_scores:
                    lines.extend(["| Metric | Mean | Std |", "|---|---|---|"])
                    for metric, scores in sorted(metric_scores.items()):
                        mean = sum(scores) / len(scores)
                        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
                        lines.append(f"| {metric} | {mean:.2f} | {std:.2f} |")
                    lines.append("")

    return "\n".join(lines)


def log_to_wandb(results: dict[str, Any], project: str = "minigpt-eval") -> None:
    """Log evaluation results to WandB as a report artifact.

    Parameters
    ----------
    results:
        Aggregated results from ``load_results``.
    project:
        WandB project name.
    """
    try:
        import wandb

        if wandb.run is None:
            wandb.init(project=project)

        # Log benchmark metrics
        for model_name, bm in results.get("benchmarks", {}).items():
            for task, task_res in bm.get("tasks", {}).items():
                if task_res.get("acc") is not None:
                    wandb.log({f"report/{model_name}/{task}/acc": task_res["acc"]})

        # Log perplexity
        for model_name, ppl in results.get("perplexity", {}).items():
            if ppl.get("perplexity") is not None:
                wandb.log({
                    f"report/{model_name}/perplexity": ppl["perplexity"],
                    f"report/{model_name}/loss": ppl.get("loss", 0),
                })

        # Log pairwise results
        for comp_name, comp in results.get("judge_pairwise", {}).items():
            wandb.log({
                f"report/pairwise/{comp_name}/win_rate_a": comp.get("win_rate_a", 0),
                f"report/pairwise/{comp_name}/win_rate_b": comp.get("win_rate_b", 0),
            })

        # Save markdown report as artifact
        md_report = generate_markdown_summary(results)
        report_path = Path("results/report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md_report)

        artifact = wandb.Artifact("evaluation_report", type="report")
        artifact.add_file(str(report_path))
        wandb.log_artifact(artifact)

        logger.info("Results logged to WandB")
    except ImportError:
        logger.warning("wandb not installed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="MiniGPT Evaluation Report Generator")
    parser.add_argument("--results-dir", required=True, help="Directory with result JSON files")
    parser.add_argument("--output", default="results/report.md", help="Output markdown path")
    parser.add_argument("--wandb", action="store_true", help="Log to WandB")
    parser.add_argument("--title", default="MiniGPT Evaluation Report")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    report = generate_markdown_summary(results, title=args.title)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    logger.info("Report written to %s", args.output)

    if args.wandb:
        log_to_wandb(results)


if __name__ == "__main__":
    main()
