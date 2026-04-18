"""CLI entry point for running benchmarks."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="awb", help="Agent Workflow Benchmark CLI")
console = Console()


def _get_workflow_map():
    """Import lazily to avoid heavy deps at import time."""
    from workflows import get_workflow_map
    return get_workflow_map()


@app.command()
def run(
    workflow: str | None = typer.Option(
        None, "--workflow", "-w", help="Run a specific workflow pattern by name"
    ),
    level: int | None = typer.Option(
        None, "--level", "-l", help="Run tasks at a specific level (1-4)"
    ),
    task_id: str | None = typer.Option(
        None, "--task-id", "-t", help="Run a single task by ID"
    ),
    no_evaluate: bool = typer.Option(
        False, "--no-evaluate", help="Skip LLM-as-judge answer quality scoring"
    ),
) -> None:
    """Run benchmark workflows against tasks."""
    from scripts.benchmark_runner import run_benchmark

    if level is not None and level not in (1, 2, 3, 4):
        console.print(f"[red]--level must be 1-4, got {level}[/red]")
        raise typer.Exit(1)

    # Validate workflow name against the live registry so the error is
    # surfaced immediately rather than silently running zero workflows.
    if workflow is not None:
        known = set(_get_workflow_map().keys())
        if workflow not in known:
            console.print(
                f"[red]Unknown workflow '{workflow}'.[/red] "
                f"Available: {', '.join(sorted(known))}"
            )
            raise typer.Exit(1)

    console.print("[bold]Agent Workflow Benchmark[/bold]")

    metrics_map = asyncio.run(
        run_benchmark(
            workflow_names=[workflow] if workflow else None,
            task_ids=[task_id] if task_id else None,
            level=level,
            evaluate=not no_evaluate,
        )
    )

    if not metrics_map:
        return

    from config import settings
    from evaluation.reporter import generate_markdown_report, summarise_to_console

    summarise_to_console(metrics_map)

    report_path = generate_markdown_report(metrics_map, settings.results_dir)
    console.print(f"  Report:  {report_path.name}")


@app.command()
def plot(
    file: str | None = typer.Option(
        None, "--file", "-f", help="Path to a specific summary JSON file (default: latest)"
    ),
    out: str = typer.Option("plots", "--out", "-o", help="Output directory for plot images"),
    show: bool = typer.Option(False, "--show", help="Show plots interactively"),
) -> None:
    """Generate benchmark visualisation plots from results."""
    from scripts.plot_results import main as _plot_main

    console.print("[bold]Generating benchmark plots...[/bold]")
    _plot_main(file=file, out=out, show=show)


@app.command()
def analyze(
    file: str | None = typer.Option(
        None, "--file", "-f", help="Path to a specific results_*.json file (default: latest)"
    ),
    out: str | None = typer.Option(
        None, "--out", "-o", help="Output directory for the analysis report (default: results/)"
    ),
) -> None:
    """Run comparative analysis: level breakdown, cost-efficiency, and insights."""
    from scripts.analyze import run_analysis

    console.print("[bold]Running comparative analysis...[/bold]")
    try:
        run_analysis(file=file, out=out)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command()
def blog(
    summary: str | None = typer.Option(
        None, "--summary", "-s", help="Path to a specific summary_*.json file (default: latest)"
    ),
    out: str | None = typer.Option(
        None, "--out", "-o", help="Output directory for the blog post (default: results/)"
    ),
) -> None:
    """Generate a publishable blog article from benchmark results."""
    from scripts.blog import run_blog

    console.print("[bold]Generating blog post...[/bold]")
    try:
        run_blog(summary_file=summary, out=out)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command()
def list_workflows() -> None:
    """List all registered workflow patterns (sourced from the live registry)."""
    # Pull descriptions from the actual workflow instances so this can never
    # drift out of sync with what the registry exposes.
    workflows = list(_get_workflow_map().values())

    table = Table(title="Workflow Patterns", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Name")
    table.add_column("Description")

    for idx, wf in enumerate(workflows, start=1):
        table.add_row(str(idx), wf.name, wf.description)

    console.print(table)


@app.command()
def list_tasks() -> None:
    """List all registered benchmark tasks."""
    from tasks import registry
    from tasks.task_registry import TaskLevel

    level_names = {
        TaskLevel.RETRIEVAL: "L1-Retrieval",
        TaskLevel.ANALYTICAL: "L2-Analytical",
        TaskLevel.REASONING: "L3-Reasoning",
        TaskLevel.DECISION: "L4-Decision",
    }

    table = Table(title="Benchmark Tasks", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Level")
    table.add_column("Description")
    table.add_column("Expected Tools")

    for task in sorted(registry.all(), key=lambda t: (t.level, t.id)):
        desc = task.description
        if len(desc) > 65:
            desc = desc[:62] + "..."
        table.add_row(
            task.id,
            level_names.get(task.level, str(task.level)),
            desc,
            ", ".join(task.expected_tools),
        )

    console.print(table)


if __name__ == "__main__":
    app()
