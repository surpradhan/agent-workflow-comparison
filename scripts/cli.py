"""CLI entry point for running benchmarks."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="awb", help="Agent Workflow Benchmark CLI")
console = Console()


@app.command()
def run(
    workflow: str | None = typer.Option(None, help="Run a specific workflow pattern by name"),
    level: int | None = typer.Option(None, help="Run tasks at a specific level (1-4)"),
    task_id: str | None = typer.Option(None, help="Run a single task by ID"),
) -> None:
    """Run benchmark workflows against tasks."""
    console.print("[bold]Agent Workflow Benchmark[/bold]")
    console.print(f"Workflow: {workflow or 'all'}")
    console.print(f"Level: {level or 'all'}")
    console.print(f"Task: {task_id or 'all'}")
    # TODO: Wire up actual benchmark runner in Week 2
    console.print("[yellow]Benchmark runner not yet implemented.[/yellow]")


@app.command()
def list_workflows() -> None:
    """List all available workflow patterns."""
    table = Table(title="Workflow Patterns")
    table.add_column("#", style="cyan")
    table.add_column("Name")
    table.add_column("Description")

    workflows = [
        ("1", "single-step", "Direct LLM call, no tools"),
        ("2", "tool-using", "LLM with tool access"),
        ("3", "chain", "Sequential multi-step pipeline"),
        ("4", "routing", "Route to specialized handlers"),
        ("5", "parallel", "Concurrent tool execution"),
        ("6", "orchestrator-workers", "Central orchestrator delegates to workers"),
        ("7", "evaluator-optimizer", "Generate-evaluate-refine loop"),
        ("8", "manager", "Hierarchical agent management"),
        ("9", "multi-agent-handoff", "Agents hand off to each other"),
        ("10", "human-in-the-loop", "Agent pauses for human input"),
    ]
    for num, name, desc in workflows:
        table.add_row(num, name, desc)

    console.print(table)


@app.command()
def list_tasks() -> None:
    """List all registered benchmark tasks."""
    # TODO: Load from task registry once tasks are defined
    console.print("[yellow]No tasks registered yet. Define tasks in tasks/.[/yellow]")


if __name__ == "__main__":
    app()
