"""Comparative analysis: per-level breakdown, cost-efficiency, and insights.

Loads raw results_*.json and produces:
  - Per-level success-rate breakdown (L1–L4) per workflow
  - Cost-efficiency metrics (quality per 1 k tokens / per second)
  - Narrative insights written to analysis_<timestamp>.md

Usage (standalone):
    python -m scripts.analyze                                    # latest file
    python -m scripts.analyze --file results/results_<ts>.json
    python -m scripts.analyze --out results/
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

LEVELS = ["L1", "L2", "L3", "L4"]
LEVEL_NAMES = {
    "L1": "Retrieval",
    "L2": "Analytical",
    "L3": "Reasoning",
    "L4": "Decision",
}

# Qualitative notes about each workflow type for the report narrative
WORKFLOW_NOTES: dict[str, str] = {
    "single_step": "Baseline — one LLM call, no tools. Fastest and cheapest; limited to tasks that fit in context.",
    "tool_using": "ReAct loop that selects and calls tools iteratively. Strong tool accuracy; latency grows with tool rounds.",
    "chain": "Four-step pipeline (decompose → retrieve → analyse → synthesise). Consistent and auditable; minimal latency overhead.",
    "routing": "Classifier dispatches to specialist handlers. Efficient for clearly typed tasks; overhead on boundary cases.",
    "parallel": "Concurrent sub-queries merged by a synthesis step. Excels at multi-angle analytical tasks; token cost multiplies.",
    "orchestrator_workers": "Central orchestrator fans out to worker agents. Best coverage on complex tasks; highest token cost.",
    "evaluator_optimizer": "Generator → Evaluator → Optimizer self-refinement loop. Highest quality ceiling; trades latency for accuracy.",
    "manager_agent": "Manager supervises specialist sub-agents. Good at decomposable decision tasks; coordination overhead visible.",
    "multi_agent_handoff": "Retrieval → Analysis → Synthesis hand-off chain. Natural fit for L3/L4 tasks; latency is additive per hop.",
    "human_in_loop": "Agentic workflow with human-approval checkpoints. Maximises correctness on high-stakes decisions; not fully automated.",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_latest_results(results_dir: Path) -> tuple[list[dict], Path]:
    """Load the most recent results_*.json from results_dir."""
    files = sorted(results_dir.glob("results_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No results_*.json files found in {results_dir}")
    return json.loads(files[0].read_text()), files[0]


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _load_matching_summary(results_file: Path) -> list[dict]:
    """Load the summary JSON whose timestamp matches the results file.

    Falls back to the most recent summary when no exact match is found,
    and warns the user so they know the metrics may be mismatched.
    """
    stem = results_file.stem  # "results_1234567890"
    ts = stem.split("_", 1)[-1]
    summary_path = results_file.parent / f"summary_{ts}.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    summaries = sorted(results_file.parent.glob("summary_*.json"), reverse=True)
    if not summaries:
        return []
    console.print(
        f"[yellow]Warning: no summary_{ts}.json found; falling back to "
        f"{summaries[0].name}. Efficiency metrics may not match this results file.[/yellow]"
    )
    return json.loads(summaries[0].read_text())


# ── Core metrics ───────────────────────────────────────────────────────────────

def _task_level(task_id: str) -> str:
    """'L1_T01' → 'L1'."""
    return task_id.split("_")[0] if "_" in task_id else "L?"


def compute_level_breakdown(results: list[dict]) -> dict[str, dict[str, dict]]:
    """Per-workflow, per-level aggregated metrics.

    Returns {workflow: {level: {success_rate, avg_tokens, avg_latency_ms, avg_quality, total}}}
    """
    raw: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {
            "success": 0, "total": 0,
            "tokens": 0, "latency": 0.0,
            "quality_sum": 0.0, "quality_count": 0,
        })
    )
    for r in results:
        lvl = _task_level(r["task_id"])
        e = raw[r["workflow_name"]][lvl]
        e["total"] += 1
        if r.get("success"):
            e["success"] += 1
        e["tokens"] += r.get("total_tokens", 0)
        e["latency"] += r.get("latency_ms", 0.0)
        if r.get("quality_score") is not None:
            e["quality_sum"] += r["quality_score"]
            e["quality_count"] += 1

    out: dict[str, dict[str, dict]] = {}
    for wf, levels in raw.items():
        out[wf] = {}
        for lvl, e in levels.items():
            n = e["total"]
            out[wf][lvl] = {
                "success_rate": e["success"] / n if n else 0.0,
                "avg_tokens": e["tokens"] / n if n else 0.0,
                "avg_latency_ms": e["latency"] / n if n else 0.0,
                "avg_quality": e["quality_sum"] / e["quality_count"] if e["quality_count"] else None,
                "total": n,
            }
    return out


def compute_cost_efficiency(summary_data: list[dict]) -> dict[str, dict]:
    """Quality-per-token and quality-per-second derived from summary metrics.

    Uses the pre-aggregated summary rather than re-deriving from raw results,
    so values are consistent with the canonical WorkflowMetrics output.
    """
    out: dict[str, dict] = {}
    for s in summary_data:
        wf = s["workflow"]
        avg_q = s.get("avg_quality_score")  # may be None
        avg_tok = s.get("avg_tokens", 0.0)
        avg_lat = s.get("avg_latency_ms", 0.0)
        out[wf] = {
            "avg_quality": avg_q,
            "avg_tokens": avg_tok,
            "avg_latency_ms": avg_lat,
            # Use explicit None checks — avg_q=0.0 is a valid (zero-quality) score
            "quality_per_1k_tokens": (avg_q / (avg_tok / 1000.0)) if (avg_q is not None and avg_tok > 0) else None,
            "quality_per_second": (avg_q / (avg_lat / 1000.0)) if (avg_q is not None and avg_lat > 0) else None,
        }
    return out


def compute_insights(
    level_breakdown: dict[str, dict[str, dict]],
    cost_efficiency: dict[str, dict],
    summary_data: list[dict],
) -> dict[str, list[str]]:
    """Return per-workflow list of notable strengths (for the report)."""
    workflows = list(level_breakdown.keys())
    strengths: dict[str, list[str]] = defaultdict(list)

    # Best at each level
    for lvl in LEVELS:
        candidates = [
            (wf, level_breakdown[wf].get(lvl, {}).get("success_rate", 0.0))
            for wf in workflows
            if lvl in level_breakdown.get(wf, {})
        ]
        if candidates:
            best_wf, best_rate = max(candidates, key=lambda x: x[1])
            if best_rate > 0:
                strengths[best_wf].append(
                    f"Best at {lvl} ({LEVEL_NAMES.get(lvl, lvl)}) tasks: {best_rate:.0%} success rate"
                )

    # Most token-efficient
    eff_candidates = [
        (wf, e["quality_per_1k_tokens"])
        for wf, e in cost_efficiency.items()
        if e.get("quality_per_1k_tokens") is not None
    ]
    if eff_candidates:
        best_wf, val = max(eff_candidates, key=lambda x: x[1])
        strengths[best_wf].append(f"Most token-efficient: {val:.4f} quality / 1k tokens")

    # Fastest
    lat_candidates = [
        (s["workflow"], s.get("avg_latency_ms", float("inf")))
        for s in summary_data
    ]
    if lat_candidates:
        fastest_wf, fastest_lat = min(lat_candidates, key=lambda x: x[1])
        strengths[fastest_wf].append(f"Fastest average latency: {fastest_lat:.0f} ms")

    # Highest overall quality
    qs_candidates = [
        (s["workflow"], s["avg_quality_score"])
        for s in summary_data
        if s.get("avg_quality_score") is not None
    ]
    if qs_candidates:
        best_wf, best_qs = max(qs_candidates, key=lambda x: x[1])
        strengths[best_wf].append(f"Highest answer quality score: {best_qs:.3f}")

    # Highest tool accuracy
    ta_candidates = [
        (s["workflow"], s.get("tool_accuracy", 0.0))
        for s in summary_data
        if s.get("tool_accuracy", 0.0) > 0
    ]
    if ta_candidates:
        best_wf, best_ta = max(ta_candidates, key=lambda x: x[1])
        strengths[best_wf].append(f"Highest tool accuracy: {best_ta:.1%}")

    return dict(strengths)


# ── Console output ─────────────────────────────────────────────────────────────

def print_level_table(level_breakdown: dict[str, dict[str, dict]]) -> None:
    workflows = sorted(level_breakdown.keys())
    levels = sorted({lvl for wf in level_breakdown.values() for lvl in wf})

    table = Table(title="Success Rate by Task Level", show_lines=True, header_style="bold cyan")
    table.add_column("Workflow", style="bold", min_width=24)
    for lvl in levels:
        table.add_column(f"{lvl} ({LEVEL_NAMES.get(lvl, '')})", justify="right")

    for wf in workflows:
        row: list[str] = [wf]
        for lvl in levels:
            sr = level_breakdown[wf].get(lvl, {}).get("success_rate")
            if sr is None:
                row.append("[dim]—[/dim]")
            elif sr >= 0.8:
                row.append(f"[green]{sr:.0%}[/green]")
            elif sr >= 0.5:
                row.append(f"[yellow]{sr:.0%}[/yellow]")
            else:
                row.append(f"[red]{sr:.0%}[/red]")
        table.add_row(*row)
    console.print(table)


def print_efficiency_table(cost_efficiency: dict[str, dict]) -> None:
    table = Table(title="Cost-Efficiency Analysis", show_lines=True, header_style="bold cyan")
    table.add_column("Workflow", style="bold", min_width=24)
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Quality / 1k Tokens", justify="right")
    table.add_column("Quality / Second", justify="right")

    for wf in sorted(cost_efficiency):
        e = cost_efficiency[wf]
        q1k = e.get("quality_per_1k_tokens")
        qps = e.get("quality_per_second")
        table.add_row(
            wf,
            f"{e['avg_tokens']:.0f}",
            f"{e['avg_latency_ms']:.0f}",
            f"{q1k:.4f}" if q1k is not None else "—",
            f"{qps:.4f}" if qps is not None else "—",
        )
    console.print(table)


# ── Recommendations ───────────────────────────────────────────────────────────

def _build_recommendations(
    level_breakdown: dict[str, dict[str, dict]],
    cost_efficiency: dict[str, dict],
    summary_data: list[dict],
) -> list[str]:
    """Build a data-driven recommendations table from actual benchmark results."""
    workflows = list(level_breakdown.keys())
    lines: list[str] = [
        "| Constraint / Use case | Best in this run | General guidance |",
        "| --- | --- | --- |",
    ]

    lat_ranked = sorted(summary_data, key=lambda s: s.get("avg_latency_ms", float("inf")))
    if lat_ranked:
        w = lat_ranked[0]
        lines.append(
            f"| Lowest latency | `{w['workflow']}` ({w.get('avg_latency_ms', 0):.0f} ms avg) | "
            "Single-pass patterns have the least coordination overhead |"
        )

    tok_ranked = sorted(summary_data, key=lambda s: s.get("avg_tokens", float("inf")))
    if tok_ranked:
        w = tok_ranked[0]
        lines.append(
            f"| Lowest token cost | `{w['workflow']}` ({w.get('avg_tokens', 0):.0f} tokens avg) | "
            "Routing and chain patterns avoid redundant LLM calls |"
        )

    qs_ranked = sorted(
        [s for s in summary_data if s.get("avg_quality_score") is not None],
        key=lambda s: s["avg_quality_score"],
        reverse=True,
    )
    if qs_ranked:
        w = qs_ranked[0]
        lines.append(
            f"| Highest answer quality | `{w['workflow']}` (score {w.get('avg_quality_score', 0):.3f}) | "
            "Self-refinement loops trade latency for accuracy |"
        )

    _level_guidance = {
        "L1": ("Simple lookups", "Single-pass or tool-using patterns are sufficient"),
        "L2": ("Analytical tasks", "Chain or parallel patterns handle multi-angle analysis"),
        "L3": ("Multi-step reasoning", "Multi-agent patterns decompose and specialise"),
        "L4": ("Decision-making", "Evaluator-optimizer or human-in-loop for high stakes"),
    }
    for lvl, (desc, guidance) in _level_guidance.items():
        candidates = [
            (wf, level_breakdown[wf].get(lvl, {}).get("success_rate", 0.0))
            for wf in workflows
            if lvl in level_breakdown.get(wf, {})
        ]
        if candidates:
            best_wf, best_sr = max(candidates, key=lambda x: x[1])
            lines.append(
                f"| {desc} ({lvl}) | `{best_wf}` ({best_sr:.0%} success) | {guidance} |"
            )

    return lines


# ── Report generation ──────────────────────────────────────────────────────────

def generate_analysis_report(
    results: list[dict],
    level_breakdown: dict[str, dict[str, dict]],
    cost_efficiency: dict[str, dict],
    insights: dict[str, list[str]],
    summary_data: list[dict],
    output_dir: Path,
) -> Path:
    """Write analysis_<ts>.md and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"analysis_{ts}.md"

    workflows = sorted({r["workflow_name"] for r in results})
    levels = sorted({_task_level(r["task_id"]) for r in results})
    total_runs = len(results)

    ranked = sorted(summary_data, key=lambda s: s.get("success_rate", 0.0), reverse=True)
    winner = ranked[0] if ranked else None

    lines: list[str] = [
        "# Comparative Analysis: Agent Workflow Benchmark",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*",
        "",
        f"**Total runs analysed:** {total_runs}  ",
        f"**Workflows compared:** {len(workflows)}  ",
        f"**Task levels:** {', '.join(f'{l} ({LEVEL_NAMES.get(l, l)})' for l in levels)}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    if winner:
        lines += [
            f"**Overall leader:** `{winner['workflow']}` "
            f"({winner.get('success_rate', 0):.1%} success rate)  ",
            "",
        ]

    lines += ["**Top 5 workflows by success rate:**", ""]
    for i, s in enumerate(ranked[:5], 1):
        lines.append(
            f"{i}. **{s['workflow']}** — "
            f"{s.get('success_rate', 0):.1%} success, "
            f"{s.get('avg_latency_ms', 0):.0f} ms avg latency, "
            f"{s.get('avg_tokens', 0):.0f} avg tokens"
        )

    lines += [
        "",
        "---",
        "",
        "## Level-by-Level Analysis",
        "",
        "Success rates broken down by task difficulty level:",
        "",
    ]

    # Table header
    header = ["Workflow"] + [f"{lvl} ({LEVEL_NAMES.get(lvl, lvl)})" for lvl in levels]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for wf in sorted(workflows):
        row = [f"`{wf}`"]
        for lvl in levels:
            sr = level_breakdown.get(wf, {}).get(lvl, {}).get("success_rate")
            row.append(f"{sr:.0%}" if sr is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "### Observations by Level", ""]
    for lvl in levels:
        candidates = [
            (wf, level_breakdown[wf].get(lvl, {}))
            for wf in workflows
            if lvl in level_breakdown.get(wf, {})
        ]
        if not candidates:
            continue
        sr_pairs = [(wf, e.get("success_rate", 0.0)) for wf, e in candidates]
        best_wf, best_sr = max(sr_pairs, key=lambda x: x[1])
        worst_wf, worst_sr = min(sr_pairs, key=lambda x: x[1])
        all_sr = [sr for _, sr in sr_pairs]
        avg_sr = sum(all_sr) / len(all_sr)

        # 95% Wilson CI half-width for the best workflow, to flag small samples
        n_tasks = candidates[0][1].get("total", 0)
        if n_tasks > 0:
            # Approximate 95% CI using normal approximation for clarity
            ci_half = 1.96 * math.sqrt(best_sr * (1 - best_sr) / n_tasks) if n_tasks > 1 else 0.5
            ci_note = f" (95% CI ±{ci_half:.0%} with n={n_tasks})"
        else:
            ci_note = ""

        lines += [
            f"**{lvl} — {LEVEL_NAMES.get(lvl, lvl)}** *(n={n_tasks} tasks per workflow)*",
            "",
            f"- Average success rate across all workflows: {avg_sr:.0%}",
            f"- Best: `{best_wf}` at {best_sr:.0%}{ci_note}",
            f"- Worst: `{worst_wf}` at {worst_sr:.0%}",
            "",
        ]

    # Statistical caveat — always shown so readers don't over-interpret small samples
    min_n = min(
        e.get("total", 0)
        for wf in workflows
        for e in level_breakdown.get(wf, {}).values()
    ) if workflows else 0
    if min_n < 10:
        lines += [
            "> **Statistical note:** With fewer than 10 tasks per level, 95% confidence "
            "intervals span ±30–49 percentage points. Differences between workflows at "
            "individual levels should be treated as indicative, not conclusive. "
            "Add more tasks per level to improve statistical power.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Cost & Efficiency Analysis",
        "",
        "| Workflow | Avg Tokens | Avg Latency (ms) | Quality / 1k Tokens | Quality / Second |",
        "| --- | --- | --- | --- | --- |",
    ]
    for wf in sorted(workflows):
        e = cost_efficiency.get(wf, {})
        tok = e.get("avg_tokens", 0)
        lat = e.get("avg_latency_ms", 0)
        q1k = e.get("quality_per_1k_tokens")
        qps = e.get("quality_per_second")
        # Format each cell independently so a missing qps doesn't hide a valid q1k
        q1k_cell = f"{q1k:.4f}" if q1k is not None else "—"
        qps_cell = f"{qps:.4f}" if qps is not None else "—"
        lines.append(f"| `{wf}` | {tok:.0f} | {lat:.0f} | {q1k_cell} | {qps_cell} |")

    lines += [
        "",
        "### Efficiency Observations",
        "",
    ]
    eff_ranked = sorted(
        [(wf, e["quality_per_1k_tokens"]) for wf, e in cost_efficiency.items() if e.get("quality_per_1k_tokens")],
        key=lambda x: x[1],
        reverse=True,
    )
    if eff_ranked:
        top_wf, top_val = eff_ranked[0]
        bot_wf, bot_val = eff_ranked[-1]
        lines += [
            f"- **Most token-efficient:** `{top_wf}` ({top_val:.4f} quality / 1k tokens)",
            f"- **Least token-efficient:** `{bot_wf}` ({bot_val:.4f} quality / 1k tokens)",
            f"- Token efficiency gap: {top_val / bot_val:.1f}× between best and worst",
            "",
        ]

    lat_ranked = sorted(
        [(s["workflow"], s.get("avg_latency_ms", 0)) for s in summary_data],
        key=lambda x: x[1],
    )
    if lat_ranked:
        fast_wf, fast_lat = lat_ranked[0]
        slow_wf, slow_lat = lat_ranked[-1]
        lines += [
            f"- **Fastest:** `{fast_wf}` ({fast_lat:.0f} ms avg)",
            f"- **Slowest:** `{slow_wf}` ({slow_lat:.0f} ms avg)",
            f"- Latency span: {slow_lat / max(fast_lat, 1):.1f}× between fastest and slowest",
            "",
        ]

    lines += [
        "---",
        "",
        "## Workflow Profiles",
        "",
    ]
    for wf in sorted(workflows):
        note = WORKFLOW_NOTES.get(wf, "")
        wf_strengths = insights.get(wf, [])
        lines += [f"### `{wf}`", ""]
        if note:
            lines += [note, ""]
        if wf_strengths:
            lines += ["**Notable strengths in this run:**", ""]
            for s in wf_strengths:
                lines.append(f"- {s}")
            lines.append("")

    lines += [
        "---",
        "",
        "## Recommendations",
        "",
        "Pattern selection guide derived from this benchmark run:",
        "",
    ]
    lines += _build_recommendations(level_breakdown, cost_efficiency, summary_data)
    lines += [""]

    path.write_text("\n".join(lines))
    return path


# ── Entry point ────────────────────────────────────────────────────────────────

def run_analysis(
    file: str | None = None,
    out: str | None = None,
) -> Path:
    """Run the full comparative analysis pipeline and return the report path.

    Raises:
        FileNotFoundError: if no results file can be found.
        ValueError:        if the results file is empty.
    """
    from config import settings

    results_dir = settings.results_dir
    output_dir = Path(out) if out else results_dir

    if file:
        results = load_results(Path(file))
        results_path = Path(file)
    else:
        results, results_path = load_latest_results(results_dir)  # raises FileNotFoundError

    if not results:
        raise ValueError(f"No results found in {results_path}")

    summary_data = _load_matching_summary(results_path)

    console.print(f"[bold]Analysing {len(results)} result(s) from {results_path.name}...[/bold]")

    level_breakdown = compute_level_breakdown(results)
    # cost_efficiency derives from summary so values match the canonical metrics
    cost_efficiency = compute_cost_efficiency(summary_data)
    insights = compute_insights(level_breakdown, cost_efficiency, summary_data)

    print_level_table(level_breakdown)
    print_efficiency_table(cost_efficiency)

    report_path = generate_analysis_report(
        results, level_breakdown, cost_efficiency, insights, summary_data, output_dir
    )
    console.print(f"\n[green]Analysis report saved:[/green] {report_path}")

    # Generate extra plots — write alongside the report, not to a CWD-relative path
    try:
        from scripts.plot_results import plot_from_raw

        plots_dir = output_dir.parent / "plots"
        written = plot_from_raw(level_breakdown, cost_efficiency, plots_dir)
        for p in written:
            console.print(f"  Plot: {p}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Skipping raw plots: {exc}[/yellow]")

    return report_path


def main(file: str | None = None, out: str | None = None) -> None:
    try:
        run_analysis(file=file, out=out)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comparative analysis of benchmark results")
    parser.add_argument("--file", help="Path to a specific results_*.json file")
    parser.add_argument("--out", help="Output directory for the analysis report")
    args = parser.parse_args()
    main(file=args.file, out=args.out)
