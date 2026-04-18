"""Core benchmark runner: runs workflows × tasks and collects metrics."""

from __future__ import annotations

import asyncio
import json
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from config import settings
from evaluation.metrics import WorkflowMetrics
from tasks import registry
from tasks.task_registry import TaskLevel
from workflows.base import WorkflowResult

console = Console()


async def run_benchmark(
    workflow_names: list[str] | None = None,
    task_ids: list[str] | None = None,
    level: int | None = None,
    evaluate: bool = True,
) -> dict[str, WorkflowMetrics]:
    """Run specified workflows against specified tasks.

    Args:
        workflow_names: Restrict to these workflow names; None = all.
        task_ids:       Restrict to these task IDs; None = all.
        level:          Restrict to tasks at this level (1–4); None = all.
        evaluate:       When True, score each answer with the LLM-as-judge
                        and populate WorkflowResult.quality_score.

    Returns a dict mapping workflow_name → WorkflowMetrics.
    """
    from workflows import get_all_workflows, get_workflow_map

    # Fresh instances for every run — avoids shared state across tasks (Fix 7)
    if workflow_names:
        workflow_map = get_workflow_map()
        workflows = [workflow_map[n] for n in workflow_names if n in workflow_map]
        unknown = [n for n in workflow_names if n not in workflow_map]
        if unknown:
            console.print(f"[yellow]Unknown workflow(s): {unknown}[/yellow]")
    else:
        workflows = get_all_workflows()

    # Select tasks
    if task_ids:
        tasks = [registry.get(tid) for tid in task_ids]
    elif level is not None:
        tasks = registry.get_by_level(TaskLevel(level))
    else:
        tasks = registry.all()

    if not workflows:
        console.print("[red]No matching workflows found.[/red]")
        return {}
    if not tasks:
        console.print("[red]No matching tasks found.[/red]")
        return {}

    console.print(
        f"Running [bold]{len(workflows)}[/bold] workflow(s) × "
        f"[bold]{len(tasks)}[/bold] task(s) = "
        f"[bold]{len(workflows) * len(tasks)}[/bold] total runs"
    )

    # Build a task_id → Task lookup for the judge phase
    task_map = {t.id: t for t in tasks}
    results_all: list[WorkflowResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for workflow in workflows:
            bar = progress.add_task(f"[cyan]{workflow.name}[/cyan]", total=len(tasks))
            for task in tasks:
                progress.update(bar, description=f"[cyan]{workflow.name}[/cyan] → {task.id}")
                result = await workflow.run(task)
                results_all.append(result)
                progress.advance(bar)

    # ── Answer quality scoring (LLM-as-judge) ─────────────────────────────
    if evaluate:
        from evaluation.judge import AnswerJudge

        judge = AnswerJudge()
        console.print("\n[bold]Scoring answer quality...[/bold]")

        async def _score_result(result: WorkflowResult) -> None:
            task = task_map.get(result.task_id)
            if task and result.answer:
                result.quality_score = await judge.score(task, str(result.answer))

        await asyncio.gather(*[_score_result(r) for r in results_all])

    # ── Aggregate metrics ──────────────────────────────────────────────────
    metrics_map: dict[str, WorkflowMetrics] = {
        w.name: WorkflowMetrics(workflow_name=w.name) for w in workflows
    }
    for result in results_all:
        metrics_map[result.workflow_name].record(result)

    _save_results(results_all, metrics_map)
    return metrics_map


def _save_results(
    results: list[WorkflowResult], metrics: dict[str, WorkflowMetrics]
) -> None:
    """Persist raw results and summary metrics to results/."""
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())

    raw_path = results_dir / f"results_{ts}.json"
    raw_path.write_text(json.dumps([r.model_dump() for r in results], indent=2, default=str))

    summary_path = results_dir / f"summary_{ts}.json"
    summary_path.write_text(json.dumps([m.summary() for m in metrics.values()], indent=2))

    console.print(f"\n[green]Results saved to {results_dir}/[/green]")
    console.print(f"  Raw:     {raw_path.name}")
    console.print(f"  Summary: {summary_path.name}")
