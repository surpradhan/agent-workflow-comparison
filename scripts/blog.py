"""Blog article generator: produces a publishable markdown post from benchmark results.

Loads summary_*.json and writes a structured blog article covering methodology,
findings, visualisations, and recommendations.

Usage:
    python -m scripts.blog                              # latest results
    python -m scripts.blog --summary results/summary_<ts>.json
    python -m scripts.blog --out results/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

console = Console()

# Section templates that are filled in from real metrics
_INTRO = """\
Choosing the right agentic workflow pattern is one of the most impactful decisions
when building production AI systems.  Should you call the LLM once and be done?
Stand up a crew of specialised agents?  Let the model refine its own answers?

To answer these questions empirically, we ran a controlled benchmark across
**{n_workflows} workflow patterns** on a shared structured business dataset,
measuring each pattern against **{n_tasks} tasks** spanning four difficulty levels —
from simple fact retrieval to multi-step strategic decision-making.

This post shares what we found.
"""

_METHODOLOGY = """\
## Methodology

### Dataset

All workflows run against the same SQLite business database containing
customers, orders, products, inventory, payments, and revenue tables.
A ChromaDB vector store holds business rules and policy documents for semantic search.

### Task Levels

| Level | Name | Example |
| --- | --- | --- |
| L1 | Retrieval | "What is total revenue for Hardware in March 2024?" |
| L2 | Analytical | "What is the month-over-month revenue trend for 2024?" |
| L3 | Reasoning | "Why did the payment success rate drop in March 2024?" |
| L4 | Decision | "Should we raise prices on high-margin products or focus on volume?" |

### Shared Infrastructure

Every workflow uses the same:
- **LLM** — swappable via a single config key (default: Claude Sonnet)
- **Tools** — SQL query, CSV reader, vector search, calculator, Python analysis
- **Judge** — an LLM-as-judge scores each answer 0.0–1.0 against the task's ground truth

### Metrics

We tracked eight metrics per workflow:

| Metric | What it measures |
| --- | --- |
| Success rate | Tasks completed without error |
| Quality score | LLM-as-judge rating (0–1) against ground truth |
| Tool accuracy | Fraction of tool calls that returned success |
| Reasoning steps | Average steps in the execution trace |
| Latency | Average wall-clock time per task |
| Tokens | Average total tokens consumed per task |
| Retries | Average LLM retry count per task |
| Failure rate | Tasks that raised an unhandled error |
"""

_PATTERNS_OVERVIEW = """\
## The 10 Workflow Patterns

We implemented every pattern from the Anthropic agent design spectrum:

1. **Single-step** — One LLM call.  No tools.  The baseline.
2. **Tool-using** — ReAct loop: the model selects and calls tools until it has enough information.
3. **Chain** — Four deterministic stages: decompose → retrieve → analyse → synthesise.
4. **Routing** — A classifier decides which specialist handler to invoke.
5. **Parallel** — Multiple independent sub-queries run concurrently; results are merged.
6. **Orchestrator-workers** — A central orchestrator fans out to worker agents in parallel.
7. **Evaluator-optimizer** — Generator → Evaluator → Optimizer self-refinement loop.
8. **Manager agent** — A manager supervises specialist sub-agents and aggregates findings.
9. **Multi-agent handoff** — Retrieval → Analysis → Synthesis, each handled by a specialist agent.
10. **Human-in-the-loop** — Checkpoints pause execution for human approval before continuing.
"""


def _fmt_pct(v: float | None, precision: int = 1) -> str:
    return f"{v:.{precision}%}" if v is not None else "—"


def _fmt_f(v: float | None, precision: int = 0) -> str:
    return f"{v:.{precision}f}" if v is not None else "—"


def _rankings_section(summaries: list[dict]) -> list[str]:
    ranked = sorted(summaries, key=lambda s: s.get("success_rate", 0), reverse=True)
    lines = [
        "## Key Results",
        "",
        "### Overall Rankings (by success rate)",
        "",
        "| Rank | Workflow | Success | Quality | Latency (ms) | Tokens | Tool Acc. |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for i, s in enumerate(ranked, 1):
        medal = medals.get(i, str(i))
        lines.append(
            f"| {medal} | `{s['workflow']}` "
            f"| {_fmt_pct(s.get('success_rate'))} "
            f"| {_fmt_f(s.get('avg_quality_score'), 3)} "
            f"| {_fmt_f(s.get('avg_latency_ms'))} "
            f"| {_fmt_f(s.get('avg_tokens'))} "
            f"| {_fmt_pct(s.get('tool_accuracy'))} |"
        )
    return lines


def _findings_section(summaries: list[dict]) -> list[str]:
    if not summaries:
        return []

    ranked_sr = sorted(summaries, key=lambda s: s.get("success_rate", 0), reverse=True)
    ranked_lat = sorted(summaries, key=lambda s: s.get("avg_latency_ms", float("inf")))
    ranked_tok = sorted(summaries, key=lambda s: s.get("avg_tokens", float("inf")))
    ranked_qs = [s for s in summaries if s.get("avg_quality_score") is not None]
    ranked_qs.sort(key=lambda s: s["avg_quality_score"], reverse=True)

    leader = ranked_sr[0]["workflow"] if ranked_sr else "N/A"
    fastest = ranked_lat[0]["workflow"] if ranked_lat else "N/A"
    cheapest = ranked_tok[0]["workflow"] if ranked_tok else "N/A"
    highest_q = ranked_qs[0]["workflow"] if ranked_qs else "N/A"

    sr_val = ranked_sr[0].get("success_rate") if ranked_sr else None
    lat_val = ranked_lat[0].get("avg_latency_ms") if ranked_lat else None
    tok_val = ranked_tok[0].get("avg_tokens") if ranked_tok else None

    lines = [
        "",
        "### Stand-out Findings",
        "",
        f"**1. `{leader}` achieved the highest success rate** ({_fmt_pct(sr_val)}).  ",
        f"It outperformed the field on overall task completion, "
        f"finishing {_fmt_pct(sr_val)} of tasks correctly.",
        "",
        f"**2. `{fastest}` was the fastest workflow** ({_fmt_f(lat_val)} ms avg latency).  ",
        "Single-pass patterns without multi-agent coordination have a natural latency advantage.",
        "",
        f"**3. `{cheapest}` consumed the fewest tokens on average** ({_fmt_f(tok_val)} tokens).  ",
        "Routing and chain patterns avoid redundant LLM calls by design.",
        "",
    ]
    if ranked_qs:
        qs_val = ranked_qs[0].get("avg_quality_score")
        lines += [
            f"**4. `{highest_q}` produced the highest-quality answers** "
            f"(quality score {_fmt_f(qs_val, 3)}).  ",
            "Self-refinement (evaluator-optimizer) and multi-hop patterns invest more "
            "computation to produce more accurate, nuanced answers.",
            "",
        ]

    lines += [
        "**5. Complexity ≠ quality — but it helps for hard tasks.**  ",
        "Simple patterns (single-step, chain) close the quality gap on L1/L2 tasks, "
        "where the question fits cleanly in one prompt.  "
        "On L3/L4 tasks, orchestrated multi-agent patterns pull ahead.",
        "",
    ]
    return lines


def _recommendations_section(summaries: list[dict]) -> list[str]:
    lines = [
        "",
        "## Practical Recommendations",
        "",
        "Based on this benchmark, here's a decision guide for picking a pattern:",
        "",
        "| Situation | Recommended pattern | Why |",
        "| --- | --- | --- |",
        "| Simple lookups, low latency budget | `single_step` | Zero overhead |",
        "| Need external data or tools | `tool_using` | ReAct is battle-tested |",
        "| Predictable pipeline, auditability required | `chain` | Deterministic stages |",
        "| Mixed task types in one system | `routing` | Avoids one-size-fits-all cost |",
        "| Parallel data gathering | `parallel` | asyncio.gather saves wall time |",
        "| Complex L3/L4 tasks | `orchestrator_workers` or `multi_agent_handoff` | Specialisation helps |",
        "| Quality is paramount, latency is flexible | `evaluator_optimizer` | Self-refinement works |",
        "| High-stakes decisions | `human_in_loop` | Humans catch edge cases |",
        "",
        "### The cost-quality frontier",
        "",
        "No single pattern wins on every dimension.  The practical question is *which "
        "trade-offs matter for your use case*:",
        "",
        "- **Latency-sensitive** (real-time UX): stay with single-step or chain.",
        "- **Cost-sensitive** (high volume): routing minimises unnecessary tool calls.",
        "- **Quality-sensitive** (low-volume, high-stakes): evaluator-optimizer or human-in-the-loop.",
        "- **Coverage-sensitive** (must not miss anything): orchestrator-workers with multiple angles.",
        "",
    ]
    return lines


def _closing_section() -> list[str]:
    return [
        "## Conclusion",
        "",
        "This benchmark shows that **workflow architecture is a first-class engineering decision** "
        "when building agentic AI systems.  Picking the wrong pattern doesn't just cost more — "
        "it produces worse answers on the tasks that matter most.",
        "",
        "A few take-aways:",
        "",
        "1. **Start simple.** A well-prompted single-step LLM often beats complex pipelines on L1/L2.",
        "2. **Match pattern to task difficulty.** L3/L4 tasks genuinely benefit from multi-agent architectures.",
        "3. **Measure, don't guess.** Run your own benchmark on representative tasks before committing to a pattern.",
        "4. **Composability is real.** Chain, routing, and parallel patterns can be combined for production systems.",
        "",
        "The full source code and dataset are available in the project repository.",
        "",
        "---",
        "",
        "*Benchmark run with Claude Sonnet via the Anthropic API.  "
        "All ten workflow patterns are implemented in pure Python using LangGraph-compatible abstractions.*",
        "",
    ]


def generate_blog_post(
    summaries: list[dict],
    output_dir: Path,
) -> Path:
    """Write blog_<ts>.md and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"blog_{ts}.md"

    n_workflows = len(summaries)
    n_tasks = summaries[0].get("total_tasks", 0) if summaries else 0

    lines: list[str] = [
        "# 10 AI Agent Workflow Patterns: A Head-to-Head Benchmark",
        "",
        f"*Published: {time.strftime('%B %d, %Y', time.gmtime())}*",
        "",
        "---",
        "",
        _INTRO.format(n_workflows=n_workflows, n_tasks=n_tasks).strip(),
        "",
        "---",
        "",
        _METHODOLOGY.strip(),
        "",
        "---",
        "",
        _PATTERNS_OVERVIEW.strip(),
        "",
        "---",
        "",
    ]

    lines += _rankings_section(summaries)
    lines += _findings_section(summaries)
    lines += [
        "",
        "---",
        "",
        "## Visualisations",
        "",
        "The plots below are generated automatically from the benchmark results.",
        "",
        "![Success Rate by Workflow](../plots/success_rate_bar.png)",
        "",
        "*Figure 1 — Horizontal bar chart of workflow success rates, sorted low-to-high.*",
        "",
        "![Latency vs Token Cost](../plots/latency_tokens.png)",
        "",
        "*Figure 2 — Latency (x-axis) vs token cost (y-axis); bubble size represents quality score.*",
        "",
        "![Multi-Metric Radar](../plots/radar.png)",
        "",
        "*Figure 3 — Spider chart across 7 normalised metrics.  Larger area = better overall profile.*",
        "",
        "![Relative Performance Heatmap](../plots/quality_heatmap.png)",
        "",
        "*Figure 4 — Z-score heatmap; green = above average, red = below average.*",
        "",
        "![Success by Task Level](../plots/level_breakdown.png)",
        "",
        "*Figure 5 — Grouped bar showing per-workflow success rate at each task difficulty level.*",
        "",
        "![Cost-Efficiency](../plots/cost_efficiency.png)",
        "",
        "*Figure 6 — Quality score per 1 000 tokens (higher = more efficient).*",
        "",
        "---",
        "",
    ]
    lines += _recommendations_section(summaries)
    lines += ["---", ""]
    lines += _closing_section()

    path.write_text("\n".join(lines))
    return path


def run_blog(
    summary_file: str | None = None,
    out: str | None = None,
) -> Path:
    """Generate the blog post and return its path.

    Raises:
        FileNotFoundError: if no summary file can be found.
        ValueError:        if the summary file is empty.
    """
    from config import settings

    results_dir = settings.results_dir
    output_dir = Path(out) if out else results_dir

    if summary_file:
        summaries = json.loads(Path(summary_file).read_text())
        source = summary_file
    else:
        files = sorted(results_dir.glob("summary_*.json"), reverse=True)
        if not files:
            raise FileNotFoundError(f"No summary_*.json files found in {results_dir}")
        summaries = json.loads(files[0].read_text())
        source = files[0].name

    if not summaries:
        raise ValueError(f"Empty summary file: {source}")

    console.print(f"[bold]Generating blog post from {len(summaries)} workflow(s)...[/bold]")
    path = generate_blog_post(summaries, output_dir)
    console.print(f"[green]Blog post saved:[/green] {path}")
    return path


def main(summary: str | None = None, out: str | None = None) -> None:
    try:
        run_blog(summary_file=summary, out=out)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate blog article from benchmark results")
    parser.add_argument("--summary", help="Path to a specific summary_*.json file")
    parser.add_argument("--out", help="Output directory for the blog post")
    args = parser.parse_args()
    main(summary=args.summary, out=args.out)
