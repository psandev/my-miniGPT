"""Google PAIR LLM Comparator integration for side-by-side model comparison.

Provides functionality to:
- Compare outputs from two model variants side-by-side
- Cluster judge rationales into themes
- Export interactive HTML comparison reports

Reference: https://github.com/PAIR-code/llm-comparator

Usage::

    python evaluation/llm_comparator.py --results results/judge_pairwise.json \\
        --output results/comparison.html
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison data builder
# ---------------------------------------------------------------------------

@staticmethod
def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_comparison_data(
    pairwise_results: dict[str, Any],
    prompts: list[str] | None = None,
    responses_a: list[str] | None = None,
    responses_b: list[str] | None = None,
) -> dict[str, Any]:
    """Build a comparison data structure from pairwise judge results.

    Parameters
    ----------
    pairwise_results:
        Results from ``vertex_judge.evaluate_pairwise``.
    prompts:
        Original prompts used for generation.
    responses_a:
        Model A's responses.
    responses_b:
        Model B's responses.

    Returns
    -------
    dict[str, Any]
        Structured comparison data suitable for visualization.
    """
    model_a = pairwise_results.get("model_a", "Model A")
    model_b = pairwise_results.get("model_b", "Model B")
    judgements = pairwise_results.get("judgements", [])

    examples = []
    for j in judgements:
        idx = j.get("prompt_index", 0)
        example = {
            "prompt": prompts[idx] if prompts and idx < len(prompts) else "",
            "response_a": responses_a[idx] if responses_a and idx < len(responses_a) else "",
            "response_b": responses_b[idx] if responses_b and idx < len(responses_b) else "",
            "winner": j.get("actual_winner", "tie"),
            "rationale": j.get("rationale", ""),
            "flipped": j.get("flipped", False),
        }
        examples.append(example)

    # Cluster rationales
    clusters = cluster_rationales([e["rationale"] for e in examples])

    return {
        "model_a": model_a,
        "model_b": model_b,
        "num_examples": len(examples),
        "win_rate_a": pairwise_results.get("win_rate_a", 0),
        "win_rate_b": pairwise_results.get("win_rate_b", 0),
        "tie_rate": pairwise_results.get("ties", 0) / max(pairwise_results.get("num_comparisons", 1), 1),
        "examples": examples,
        "rationale_clusters": clusters,
    }


def cluster_rationales(
    rationales: list[str],
    n_clusters: int = 5,
) -> list[dict[str, Any]]:
    """Cluster judge rationales into thematic groups.

    Uses simple keyword-based clustering as a lightweight alternative
    to full topic modeling. For production, consider using embedding-based
    clustering via sentence-transformers.

    Parameters
    ----------
    rationales:
        List of rationale strings from the judge.
    n_clusters:
        Target number of clusters.

    Returns
    -------
    list[dict[str, Any]]
        List of cluster dicts with ``theme``, ``count``, and ``examples``.
    """
    # Simple keyword-based theme extraction
    theme_keywords = {
        "coherence": ["coherent", "logical", "structured", "organized", "flow"],
        "detail": ["detailed", "specific", "thorough", "comprehensive", "elaborate"],
        "accuracy": ["accurate", "correct", "factual", "precise", "right"],
        "fluency": ["fluent", "natural", "readable", "smooth", "well-written"],
        "helpfulness": ["helpful", "informative", "useful", "practical", "relevant"],
        "creativity": ["creative", "original", "imaginative", "novel", "engaging"],
        "safety": ["safe", "appropriate", "responsible", "harmful", "toxic"],
        "clarity": ["clear", "concise", "understandable", "simple", "straightforward"],
    }

    clusters: dict[str, list[str]] = {theme: [] for theme in theme_keywords}
    clusters["other"] = []

    for rationale in rationales:
        if not rationale:
            continue
        rationale_lower = rationale.lower()
        matched = False
        for theme, keywords in theme_keywords.items():
            if any(kw in rationale_lower for kw in keywords):
                clusters[theme].append(rationale)
                matched = True
                break
        if not matched:
            clusters["other"].append(rationale)

    # Build output, sorted by count, limited to n_clusters
    result = []
    for theme, examples in sorted(clusters.items(), key=lambda x: -len(x[1])):
        if not examples:
            continue
        result.append({
            "theme": theme,
            "count": len(examples),
            "examples": examples[:3],  # Top 3 examples per cluster
        })
        if len(result) >= n_clusters:
            break

    return result


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def export_html(
    comparison_data: dict[str, Any],
    output_path: str = "results/comparison.html",
) -> str:
    """Export comparison data as an interactive HTML report.

    Parameters
    ----------
    comparison_data:
        Data from ``build_comparison_data``.
    output_path:
        File path for the HTML output.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    model_a = comparison_data["model_a"]
    model_b = comparison_data["model_b"]
    win_a = comparison_data["win_rate_a"]
    win_b = comparison_data["win_rate_b"]
    tie_rate = comparison_data["tie_rate"]
    examples = comparison_data["examples"]
    clusters = comparison_data.get("rationale_clusters", [])

    # Build HTML
    examples_html = ""
    for i, ex in enumerate(examples):
        winner_class = "winner-a" if ex["winner"] == model_a else ("winner-b" if ex["winner"] == model_b else "tie")
        examples_html += f"""
        <div class="example {winner_class}">
            <h3>Example {i + 1} — Winner: {ex['winner']}</h3>
            <div class="prompt"><strong>Prompt:</strong> {_escape_html(ex['prompt'][:500])}</div>
            <div class="responses">
                <div class="response-a">
                    <h4>{model_a}</h4>
                    <p>{_escape_html(ex['response_a'][:500])}</p>
                </div>
                <div class="response-b">
                    <h4>{model_b}</h4>
                    <p>{_escape_html(ex['response_b'][:500])}</p>
                </div>
            </div>
            <div class="rationale"><strong>Rationale:</strong> {_escape_html(ex.get('rationale', '')[:300])}</div>
        </div>
        """

    clusters_html = ""
    for cluster in clusters:
        clusters_html += f"""
        <div class="cluster">
            <h4>{cluster['theme'].title()} ({cluster['count']} mentions)</h4>
            <ul>{''.join(f"<li>{_escape_html(e[:200])}</li>" for e in cluster['examples'])}</ul>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LLM Comparison: {model_a} vs {model_b}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .bar {{ display: flex; height: 30px; border-radius: 4px; overflow: hidden; margin: 10px 0; }}
        .bar-a {{ background: #4CAF50; }}
        .bar-b {{ background: #2196F3; }}
        .bar-tie {{ background: #9E9E9E; }}
        .example {{ border: 1px solid #ddd; padding: 16px; margin: 12px 0; border-radius: 8px; }}
        .winner-a {{ border-left: 4px solid #4CAF50; }}
        .winner-b {{ border-left: 4px solid #2196F3; }}
        .tie {{ border-left: 4px solid #9E9E9E; }}
        .responses {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }}
        .response-a, .response-b {{ padding: 12px; background: #fafafa; border-radius: 4px; }}
        .rationale {{ color: #666; font-style: italic; margin-top: 8px; }}
        .cluster {{ margin: 8px 0; padding: 12px; background: #f9f9f9; border-radius: 4px; }}
        .prompt {{ margin: 8px 0; color: #333; }}
    </style>
</head>
<body>
    <h1>LLM Comparison Report</h1>
    <div class="summary">
        <h2>{model_a} vs {model_b}</h2>
        <p>{comparison_data['num_examples']} comparisons</p>
        <div class="bar">
            <div class="bar-a" style="width: {win_a * 100:.1f}%;" title="{model_a}: {win_a * 100:.1f}%"></div>
            <div class="bar-tie" style="width: {tie_rate * 100:.1f}%;" title="Tie: {tie_rate * 100:.1f}%"></div>
            <div class="bar-b" style="width: {win_b * 100:.1f}%;" title="{model_b}: {win_b * 100:.1f}%"></div>
        </div>
        <p>{model_a}: {win_a * 100:.1f}% | Tie: {tie_rate * 100:.1f}% | {model_b}: {win_b * 100:.1f}%</p>
    </div>

    <h2>Rationale Themes</h2>
    {clusters_html}

    <h2>Examples</h2>
    {examples_html}
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    logger.info("HTML comparison report exported to %s", output_path)
    return output_path


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="LLM Comparator - Side-by-side comparison")
    parser.add_argument("--results", required=True, help="Pairwise judge results JSON")
    parser.add_argument("--responses", type=str, default=None, help="Responses JSON with prompts/responses")
    parser.add_argument("--output", default="results/comparison.html", help="HTML output path")
    args = parser.parse_args()

    with open(args.results) as f:
        pairwise_results = json.load(f)

    prompts = None
    responses_a = None
    responses_b = None
    if args.responses:
        with open(args.responses) as f:
            resp_data = json.load(f)
            prompts = resp_data.get("prompts")
            responses_a = resp_data.get("responses_a")
            responses_b = resp_data.get("responses_b")

    comparison_data = build_comparison_data(
        pairwise_results, prompts, responses_a, responses_b
    )
    export_html(comparison_data, args.output)


if __name__ == "__main__":
    main()
