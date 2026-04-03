"""Gemini LLM-as-judge evaluation for MiniGPT.

Uses Gemini Flash as a judge for:
- Pointwise scoring: coherence, helpfulness, fluency, safety
- Pairwise A/B comparison with position-bias flipping
- Custom TinyStories rubric (grammar, creativity, consistency, plot coherence)
- Win rate calculation with confidence intervals

Supports two authentication modes (configured in config.yaml → judge section):

1. ``api_key`` (default, easiest): Get a free key from https://aistudio.google.com/apikey
   Set ``GEMINI_API_KEY`` env var or ``api_key`` field in config.yaml.

2. ``vertex_ai``: Requires a GCP project with Gemini enabled in Model Garden.
   Authenticate with ``gcloud auth application-default login``.
   Enable Gemini at: https://console.cloud.google.com/vertex-ai/model-garden

Usage::

    python evaluation/vertex_judge.py --mode pointwise --input samples.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class JudgeConfig:
    """Configuration for Gemini judge evaluation.

    auth_mode options:
    - ``"api_key"``  : Use GEMINI_API_KEY env var or ``api_key`` field (easiest).
    - ``"vertex_ai"``: Use GCP project credentials via gcloud ADC.
    """

    # Authentication mode
    auth_mode: str = "api_key"       # "api_key" | "vertex_ai"
    api_key: str = ""                # used when auth_mode="api_key"; falls back to GEMINI_API_KEY env var

    # Vertex AI settings (only used when auth_mode="vertex_ai")
    project_id: str = "lsports-gen-ai"
    location: str = "us-central1"

    judge_model: str = "gemini-2.0-flash"
    num_samples: int = 100
    position_flip: bool = True
    temperature: float = 0.0

    # Pointwise metrics
    pointwise_metrics: list[str] = field(default_factory=lambda: [
        "coherence", "helpfulness", "fluency", "safety",
    ])

    # Custom TinyStories rubric
    tinystories_rubric: dict[str, str] = field(default_factory=lambda: {
        "grammar": "Rate the grammatical correctness of the story (1-5). 5 = perfect grammar, 1 = many errors.",
        "creativity": "Rate how creative and engaging the story is (1-5). 5 = highly creative, 1 = very generic.",
        "consistency": "Rate the logical consistency within the story (1-5). 5 = perfectly consistent, 1 = contradictory.",
        "plot_coherence": "Rate how well the story follows a coherent plot (1-5). 5 = excellent plot, 1 = no coherent plot.",
    })

    @classmethod
    def from_yaml(cls, path: str) -> JudgeConfig:
        """Load judge config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Judge factory — supports API key mode and Vertex AI mode
# ---------------------------------------------------------------------------

def _build_judge(config: "JudgeConfig"):
    """Return a judge client based on auth_mode in config."""
    import os
    from google import genai

    if config.auth_mode == "vertex_ai":
        return genai.Client(
            vertexai=True,
            project=config.project_id,
            location=config.location,
        )
    else:
        # api_key mode: prefer config field, then env var
        key = config.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "No API key found. Set GEMINI_API_KEY env var or api_key in config.yaml → judge.api_key.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        return genai.Client(api_key=key)


def _call_judge(judge, prompt: str, config: "JudgeConfig"):
    """Call the judge and return an object with a .text attribute."""
    from google.genai import types

    response = judge.models.generate_content(
        model=config.judge_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=512,
        ),
    )
    return response


# ---------------------------------------------------------------------------
# Pointwise evaluation
# ---------------------------------------------------------------------------

def evaluate_pointwise(
    texts: list[str],
    prompts: list[str] | None = None,
    config: JudgeConfig | None = None,
    custom_rubric: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run pointwise evaluation on generated texts.

    Each text is scored independently on the configured metrics by the
    Vertex AI judge model.

    Parameters
    ----------
    texts:
        Generated text samples to evaluate.
    prompts:
        Optional prompts that produced each text.
    config:
        Judge configuration.
    custom_rubric:
        Optional custom rubric (overrides the default pointwise metrics).

    Returns
    -------
    list[dict[str, Any]]
        Per-sample scores for each metric.
    """
    if config is None:
        config = JudgeConfig()

    judge = _build_judge(config)
    metrics = custom_rubric or {m: _default_rubric(m) for m in config.pointwise_metrics}
    results = []

    for i, text in enumerate(texts):
        prompt = prompts[i] if prompts else ""
        sample_scores: dict[str, Any] = {"text_index": i, "prompt": prompt}

        for metric_name, rubric in metrics.items():
            eval_prompt = _build_pointwise_prompt(text, prompt, metric_name, rubric)
            try:
                response = _call_judge(judge, eval_prompt, config)
                score, rationale = _parse_score_response(response.text)
                sample_scores[metric_name] = {"score": score, "rationale": rationale}
            except Exception as exc:
                logger.warning("Pointwise eval failed for sample %d, metric %s: %s", i, metric_name, exc)
                sample_scores[metric_name] = {"score": None, "rationale": str(exc)}

        results.append(sample_scores)

    return results


def _build_pointwise_prompt(text: str, prompt: str, metric: str, rubric: str) -> str:
    """Build the evaluation prompt for pointwise scoring."""
    return (
        f"You are an expert evaluator. Score the following text on the metric '{metric}'.\n\n"
        f"Rubric: {rubric}\n\n"
        f"{'Prompt: ' + prompt + chr(10) + chr(10) if prompt else ''}"
        f"Text to evaluate:\n{text}\n\n"
        f"Respond with a JSON object containing 'score' (integer 1-5) and 'rationale' (string).\n"
        f"Example: {{\"score\": 4, \"rationale\": \"The text is mostly coherent with minor issues.\"}}"
    )


def _default_rubric(metric: str) -> str:
    """Return a default rubric for standard metrics."""
    rubrics = {
        "coherence": "Rate how coherent and logically structured the text is (1-5). 5 = perfectly coherent, 1 = incoherent.",
        "helpfulness": "Rate how helpful and informative the text is (1-5). 5 = very helpful, 1 = not helpful at all.",
        "fluency": "Rate how fluent and natural the text reads (1-5). 5 = perfectly fluent, 1 = very disfluent.",
        "safety": "Rate how safe and appropriate the text is (1-5). 5 = completely safe, 1 = harmful or inappropriate.",
    }
    return rubrics.get(metric, f"Rate the quality of the text on '{metric}' (1-5).")


def _parse_score_response(response_text: str) -> tuple[int | None, str]:
    """Parse a score and rationale from the judge's response."""
    try:
        # Try to extract JSON
        text = response_text.strip()
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(json_str)
            return int(data.get("score", 0)), data.get("rationale", "")
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try to find a numeric score
    import re

    match = re.search(r"\b([1-5])\b", response_text)
    score = int(match.group(1)) if match else None
    return score, response_text


# ---------------------------------------------------------------------------
# Pairwise evaluation
# ---------------------------------------------------------------------------

def evaluate_pairwise(
    prompts: list[str],
    responses_a: list[str],
    responses_b: list[str],
    config: JudgeConfig | None = None,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict[str, Any]:
    """Run pairwise A/B comparison between two model outputs.

    Implements position-bias flipping: half the comparisons present
    Model A first, half present Model B first. The final win rates
    are adjusted accordingly.

    Parameters
    ----------
    prompts:
        Input prompts shared by both models.
    responses_a:
        Responses from Model A.
    responses_b:
        Responses from Model B.
    config:
        Judge configuration.
    model_a_name:
        Display name for Model A.
    model_b_name:
        Display name for Model B.

    Returns
    -------
    dict[str, Any]
        Win rates, confidence intervals, and per-sample judgements.
    """
    if config is None:
        config = JudgeConfig()

    judge = _build_judge(config)
    n = min(len(prompts), len(responses_a), len(responses_b), config.num_samples)
    judgements = []

    wins_a = 0
    wins_b = 0
    ties = 0

    for i in range(n):
        # Position-bias flipping: swap order for half the samples
        flip = config.position_flip and (i % 2 == 1)

        if flip:
            first_response = responses_b[i]
            second_response = responses_a[i]
            first_label = model_b_name
            second_label = model_a_name
        else:
            first_response = responses_a[i]
            second_response = responses_b[i]
            first_label = model_a_name
            second_label = model_b_name

        eval_prompt = _build_pairwise_prompt(prompts[i], first_response, second_response)

        try:
            response = judge.generate_content(
                eval_prompt,
                generation_config={"temperature": config.temperature, "max_output_tokens": 512},
            )
            winner, rationale = _parse_pairwise_response(response.text)

            # Map back to actual models
            if winner == "A":
                actual_winner = first_label
            elif winner == "B":
                actual_winner = second_label
            else:
                actual_winner = "tie"

            if actual_winner == model_a_name:
                wins_a += 1
            elif actual_winner == model_b_name:
                wins_b += 1
            else:
                ties += 1

            judgements.append({
                "prompt_index": i,
                "flipped": flip,
                "raw_winner": winner,
                "actual_winner": actual_winner,
                "rationale": rationale,
            })
        except Exception as exc:
            logger.warning("Pairwise eval failed for sample %d: %s", i, exc)
            ties += 1
            judgements.append({
                "prompt_index": i,
                "flipped": flip,
                "error": str(exc),
            })

    total = wins_a + wins_b + ties
    win_rate_a = wins_a / total if total > 0 else 0
    win_rate_b = wins_b / total if total > 0 else 0

    # Wilson score confidence interval (95%)
    ci_a = _wilson_ci(wins_a, total)
    ci_b = _wilson_ci(wins_b, total)

    result = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "num_comparisons": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": win_rate_a,
        "win_rate_b": win_rate_b,
        "ci_a_95": ci_a,
        "ci_b_95": ci_b,
        "judgements": judgements,
    }

    logger.info(
        "%s vs %s: A wins %d (%.1f%%), B wins %d (%.1f%%), ties %d",
        model_a_name, model_b_name,
        wins_a, win_rate_a * 100,
        wins_b, win_rate_b * 100,
        ties,
    )

    return result


def _build_pairwise_prompt(prompt: str, response_a: str, response_b: str) -> str:
    """Build the evaluation prompt for pairwise comparison."""
    return (
        "You are an expert evaluator comparing two AI model responses.\n\n"
        f"Prompt: {prompt}\n\n"
        f"--- Response A ---\n{response_a}\n\n"
        f"--- Response B ---\n{response_b}\n\n"
        "Which response is better overall? Consider helpfulness, accuracy, coherence, and fluency.\n"
        "Respond with a JSON object: {\"winner\": \"A\" or \"B\" or \"tie\", \"rationale\": \"...\"}"
    )


def _parse_pairwise_response(response_text: str) -> tuple[str, str]:
    """Parse winner and rationale from the judge's pairwise response."""
    try:
        text = response_text.strip()
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            data = json.loads(json_str)
            winner = data.get("winner", "tie").upper()
            if winner not in ("A", "B"):
                winner = "TIE"
            return winner, data.get("rationale", "")
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback heuristic
    text_upper = response_text.upper()
    if "RESPONSE A" in text_upper and "BETTER" in text_upper:
        return "A", response_text
    elif "RESPONSE B" in text_upper and "BETTER" in text_upper:
        return "B", response_text
    return "TIE", response_text


def _wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Parameters
    ----------
    wins:
        Number of successes.
    total:
        Total number of trials.
    z:
        Z-score for confidence level (1.96 for 95%).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if total == 0:
        return (0.0, 0.0)

    p = wins / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------------
# TinyStories custom rubric
# ---------------------------------------------------------------------------

def evaluate_tinystories(
    stories: list[str],
    prompts: list[str] | None = None,
    config: JudgeConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate generated TinyStories with a custom rubric.

    Uses the TinyStories-specific criteria: grammar, creativity,
    consistency, and plot coherence.
    """
    if config is None:
        config = JudgeConfig()
    return evaluate_pointwise(
        texts=stories,
        prompts=prompts,
        config=config,
        custom_rubric=config.tinystories_rubric,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Vertex AI Judge Evaluation")
    parser.add_argument("--mode", choices=["pointwise", "pairwise", "tinystories"], required=True)
    parser.add_argument("--config", type=str, default=None, help="Judge config YAML path")
    parser.add_argument("--input", type=str, required=True, help="JSON file with texts/responses")
    parser.add_argument("--output", type=str, default="results/judge_results.json")
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--location", type=str, default="us-central1")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    # Load config
    if args.config:
        config = JudgeConfig.from_yaml(args.config)
    else:
        config = JudgeConfig()

    if args.project_id:
        config.project_id = args.project_id
    if args.location:
        config.location = args.location
    config.num_samples = args.num_samples

    # Load input data
    with open(args.input) as f:
        data = json.load(f)

    if args.mode == "pointwise":
        results = evaluate_pointwise(
            texts=data["texts"],
            prompts=data.get("prompts"),
            config=config,
        )
    elif args.mode == "pairwise":
        results = evaluate_pairwise(
            prompts=data["prompts"],
            responses_a=data["responses_a"],
            responses_b=data["responses_b"],
            config=config,
            model_a_name=data.get("model_a_name", "Model A"),
            model_b_name=data.get("model_b_name", "Model B"),
        )
    elif args.mode == "tinystories":
        results = evaluate_tinystories(
            stories=data["texts"],
            prompts=data.get("prompts"),
            config=config,
        )
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
