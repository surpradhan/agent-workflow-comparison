"""Evaluation reporter: generates comparison tables and markdown reports.

Takes the WorkflowMetrics map produced by benchmark_runner and renders:
  - A Rich console table for immediate display
  - A markdown report written to results/report_<timestamp>.md
  - A structured dict/JSON suitable for further analysis

Ranking logic: workflows are ranked by a composite score that weights
success rate (40%), quality score (30%), and penalises latency and tokens.
When no quality scores are available, success rate carries 70% of the weight.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evaluation.metrics import WorkflowMetrics

console = Console()

# Metric display configuration: (attr, label, format, higher_is_better)
_METRICS: list[tuple[str, str, str, bool]] = [
    ("success_rate",        "Success %",     ".1%",  True),
    ("avg_quality_score",   "Quality",       ".3f",  True),
    ("tool_accuracy",       "Tool Acc.",     ".1%",  True),
    ("avg_reasoning_steps", "Steps",         ".1f",  True),
    ("avg_latency_ms",      "Latency (ms)",  ".0f",  False),
    ("avg_tokens",          "Tokens",        ".0f",  False),
    ("avg_retries",         "Retries",       ".2f",  False),
    ("failure_rate",        "Fail %",        ".1%",  False),
]


def _normalization_context(metrics: dict[str, WorkflowMetrics]) -> dict[str, tuple[float, float]]:
    """Compute min/max ranges for latency and tokens across all workflows.

    Used by composite_score so efficiency penalties are scaled to this run,
    not raw milliseconds/tokens (which vary wildly by hardware and model).
    """
    all_m = list(metrics.values())
    lat_vals = [m.avg_latency_ms for m in all_m]
    tok_vals = [m.avg_tokens for m in all_m]
    return {
        "latency": (min(lat_vals), max(lat_vals)),
        "tokens":  (min(tok_vals), max(tok_vals)),
    }


def composite_score(
    m: WorkflowMetrics,
    ctx: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Single comparable score — higher is better.

    Formula (when quality scores are available):
      0.40 × success_rate
    + 0.30 × avg_quality_score
    + 0.15 × tool_accuracy
    - 0.10 × norm_latency      (0–1, min-max scaled within the run)
    - 0.05 × norm_tokens

    Without quality scores:
      0.70 × success_rate + 0.30 × tool_accuracy − efficiency penalties

    Args:
        m:   The workflow's aggregated metrics.
        ctx: Cross-workflow normalization ranges from _normalization_context().
             When None, efficiency penalties are omitted (useful for single-
             workflow display where cross-run scaling is unavailable).
    """
    norm_lat = norm_tok = 0.0
    if ctx:
        lat_lo, lat_hi = ctx["latency"]
        tok_lo, tok_hi = ctx["tokens"]
        lat_span = lat_hi - lat_lo
        tok_span = tok_hi - tok_lo
        norm_lat = (m.avg_latency_ms - lat_lo) / lat_span if lat_span > 0 else 0.0
        norm_tok = (m.avg_tokens - tok_lo) / tok_span if tok_span > 0 else 0.0

    efficiency_penalty = 0.10 * norm_lat + 0.05 * norm_tok

    qs = m.avg_quality_score
    if qs is not None:
        return 0.40 * m.success_rate + 0.30 * qs + 0.15 * m.tool_accuracy - efficiency_penalty
    return 0.70 * m.success_rate + 0.30 * m.tool_accuracy - efficiency_penalty


def rank_workflows(metrics: dict[str, WorkflowMetrics]) -> list[WorkflowMetrics]:
    """Return workflows sorted best-first by composite score (with efficiency penalties)."""
    ctx = _normalization_context(metrics)
    return sorted(metrics.values(), key=lambda m: composite_score(m, ctx), reverse=True)


def print_comparison_table(metrics: dict[str, WorkflowMetrics]) -> None:
    """Render a Rich comparison table to the console."""
    ranked = rank_workflows(metrics)

    table = Table(
        title="Workflow Benchmark Results",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Rank", style="bold", justify="center", width=5)
    table.add_column("Workflow", style="bold", min_width=24)
    table.add_column("Tasks", justify="right", width=6)

    for _, label, _, _ in _METRICS:
        table.add_column(label, justify="right")

    # Precompute best/worst values for highlighting
    best: dict[str, float] = {}
    worst: dict[str, float] = {}
    for attr, _, _, higher_is_better in _METRICS:
        vals = []
        for m in metrics.values():
            v = getattr(m, attr, None)
            if v is not None:
                vals.append(v)
        if vals:
            best[attr] = max(vals) if higher_is_better else min(vals)
            worst[attr] = min(vals) if higher_is_better else max(vals)

    for rank, m in enumerate(ranked, start=1):
        cells: list[str] = []
        for attr, _, fmt, higher_is_better in _METRICS:
            val = getattr(m, attr, None)
            if val is None:
                cells.append("[dim]—[/dim]")
                continue
            formatted = format(val, fmt)
            if attr in best and val == best[attr]:
                cells.append(f"[bold green]{formatted}[/bold green]")
            elif attr in worst and val == worst[attr]:
                cells.append(f"[red]{formatted}[/red]")
            else:
                cells.append(formatted)

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
        table.add_row(medal, m.workflow_name, str(m.total_tasks), *cells)

    console.print(table)


def generate_markdown_report(
    metrics: dict[str, WorkflowMetrics],
    output_dir: Path,
) -> Path:
    """Write a markdown benchmark report and return its path."""
    ctx = _normalization_context(metrics)
    ranked = rank_workflows(metrics)
    ts = int(time.time())
    report_path = output_dir / f"report_{ts}.md"

    lines: list[str] = [
        "# Agent Workflow Benchmark Report",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*",
        "",
        f"**Workflows evaluated:** {len(metrics)}  ",
        f"**Tasks per workflow:** {next(iter(metrics.values())).total_tasks if metrics else 0}",
        "",
        "---",
        "",
        "## Rankings",
        "",
        "Workflows are ranked by composite score:  ",
        "`0.40 × success_rate + 0.30 × quality_score + 0.15 × tool_accuracy`  ",
        "`- 0.10 × norm_latency - 0.05 × norm_tokens`  ",
        "(latency and tokens are min-max normalised across this run; "
        "quality score omitted from formula when unavailable)",
        "",
    ]

    for rank, m in enumerate(ranked, start=1):
        score = composite_score(m, ctx)
        lines.append(f"{rank}. **{m.workflow_name}** — composite score `{score:.4f}`")

    lines += [
        "",
        "---",
        "",
        "## Full Metrics Table",
        "",
    ]

    # Table header
    header_cols = ["Workflow", "Tasks"] + [label for _, label, _, _ in _METRICS]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for m in ranked:
        row = [m.workflow_name, str(m.total_tasks)]
        for attr, _, fmt, _ in _METRICS:
            val = getattr(m, attr, None)
            row.append(format(val, fmt) if val is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Definition |",
        "| --- | --- |",
        "| Success % | Fraction of tasks completed without error |",
        "| Quality | LLM-as-judge score 0–1 (omitted if no ground truth) |",
        "| Tool Acc. | Mean per-task fraction of tool calls returning `success=True` |",
        "| Steps | Average reasoning steps per task |",
        "| Latency (ms) | Average wall-clock time per task |",
        "| Tokens | Average total tokens consumed per task |",
        "| Retries | Average LLM retry count per task |",
        "| Fail % | Fraction of tasks that raised an error |",
        "",
        "---",
        "",
        "## Per-Workflow Summaries",
        "",
    ]

    for m in ranked:
        lines += [
            f"### {m.workflow_name}",
            "",
            f"- **Total tasks:** {m.total_tasks}",
            f"- **Success rate:** {m.success_rate:.1%}",
            f"- **Composite score:** {composite_score(m, ctx):.4f}",
        ]
        qs = m.avg_quality_score
        if qs is not None:
            lines.append(f"- **Avg quality score:** {qs:.3f}")
        lines += [
            f"- **Tool accuracy:** {m.tool_accuracy:.1%}",
            f"- **Avg latency:** {m.avg_latency_ms:.0f} ms",
            f"- **Avg tokens:** {m.avg_tokens:.0f}",
            f"- **Avg retries:** {m.avg_retries:.2f}",
            "",
        ]

    report_path.write_text("\n".join(lines))
    return report_path


def summarise_to_console(metrics: dict[str, WorkflowMetrics]) -> None:
    """Print table + winner callout to the console."""
    print_comparison_table(metrics)
    ctx = _normalization_context(metrics)
    ranked = rank_workflows(metrics)
    if ranked:
        winner = ranked[0]
        console.print(
            f"\n[bold green]Winner:[/bold green] [bold]{winner.workflow_name}[/bold] "
            f"(composite score {composite_score(winner, ctx):.4f})"
        )
