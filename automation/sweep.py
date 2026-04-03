"""WandB Sweep launcher for Bayesian hyperparameter optimization.

Searches over learning rate, warmup steps, batch size, and weight decay
to minimize validation perplexity.

Usage::

    python automation/sweep.py --config configs/sweep_config.yaml \\
        --preset small --count 20
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.model.config import ModelConfig, TrainingConfig, PRESETS

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_perplexity", "goal": "minimize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "warmup_steps": {"values": [500, 1000, 2000, 4000]},
        "batch_size": {"values": [4, 8, 16, 32]},
        "weight_decay": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.3},
    },
}


def load_sweep_config(config_path: str) -> dict[str, Any]:
    """Load sweep configuration from a YAML file."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_sweep_trial() -> None:
    """Run a single sweep trial.

    Called by WandB's sweep agent. Reads hyperparameters from ``wandb.config``
    and trains a model, logging validation perplexity as the optimization metric.
    """
    import wandb

    from training.train import train
    from evaluation.perplexity import evaluate_perplexity

    run = wandb.init()
    config = wandb.config

    # Get model preset from sweep config
    preset = config.get("preset", "small")
    model_config = copy.deepcopy(PRESETS.get(preset, ModelConfig()))

    # Build training config from sweep hyperparameters
    training_config = TrainingConfig(
        lr=config.get("lr", 3e-4),
        warmup_steps=config.get("warmup_steps", 2000),
        batch_size=config.get("batch_size", 8),
        weight_decay=config.get("weight_decay", 0.1),
        max_steps=config.get("max_steps", 5000),
    )

    dataset_name = config.get("dataset", "smollm")
    tokenizer_name = config.get("tokenizer", "meta-llama/Llama-3.2-1B")

    # Train
    output_dir = f"sweep_runs/{run.id}"
    checkpoint_dir = train(
        model_config=model_config,
        training_config=training_config,
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        use_wandb=False,  # Sweep agent manages WandB
    )

    # Evaluate perplexity
    checkpoint_path = str(checkpoint_dir / "checkpoint.pt")
    try:
        ppl_result = evaluate_perplexity(
            checkpoint_path=checkpoint_path,
            dataset_name="wikitext2",
            tokenizer_name=tokenizer_name,
        )
        val_ppl = ppl_result["perplexity"]
    except Exception as exc:
        logger.warning("Perplexity evaluation failed: %s", exc)
        val_ppl = float("inf")

    wandb.log({"val_perplexity": val_ppl})
    logger.info("Sweep trial complete: val_perplexity=%.2f", val_ppl)


def launch_sweep(
    sweep_config: dict[str, Any] | None = None,
    config_path: str | None = None,
    preset: str = "small",
    dataset: str = "smollm",
    max_steps: int = 5000,
    count: int = 20,
    project: str = "minigpt-sweep",
) -> str:
    """Launch a WandB sweep.

    Parameters
    ----------
    sweep_config:
        Sweep configuration dict. If ``None``, loads from ``config_path``
        or uses defaults.
    config_path:
        Path to sweep config YAML.
    preset:
        Model preset to optimize.
    dataset:
        Training dataset.
    max_steps:
        Maximum training steps per trial.
    count:
        Number of sweep trials.
    project:
        WandB project name.

    Returns
    -------
    str
        The WandB sweep ID.
    """
    import wandb

    if sweep_config is None:
        if config_path:
            sweep_config = load_sweep_config(config_path)
        else:
            sweep_config = copy.deepcopy(DEFAULT_SWEEP_CONFIG)

    # Inject fixed parameters
    sweep_config.setdefault("parameters", {})
    sweep_config["parameters"]["preset"] = {"value": preset}
    sweep_config["parameters"]["dataset"] = {"value": dataset}
    sweep_config["parameters"]["max_steps"] = {"value": max_steps}

    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project)
    logger.info("Created WandB sweep: %s (project: %s)", sweep_id, project)

    # Launch agent
    logger.info("Launching sweep agent for %d trials", count)
    wandb.agent(sweep_id, function=run_sweep_trial, count=count, project=project)

    return sweep_id


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="MiniGPT WandB Sweep Launcher")
    parser.add_argument("--config", default=None, help="Sweep config YAML path")
    parser.add_argument("--preset", default="small", help=f"Model preset: {list(PRESETS.keys())}")
    parser.add_argument("--dataset", default="smollm")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--count", type=int, default=20, help="Number of sweep trials")
    parser.add_argument("--project", default="minigpt-sweep")
    args = parser.parse_args()

    launch_sweep(
        config_path=args.config,
        preset=args.preset,
        dataset=args.dataset,
        max_steps=args.max_steps,
        count=args.count,
        project=args.project,
    )


if __name__ == "__main__":
    main()
