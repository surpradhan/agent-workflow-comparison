"""Evaluation metrics for comparing workflow performance.

Tracks all 7 metrics from the benchmark blueprint:
task success rate, tool accuracy, reasoning steps, latency, token cost,
retries, and failure rate.  Answer quality (LLM-as-judge) is tracked as
an optional 8th metric when ground truth is available.

Running-sum design: WorkflowResult objects are NOT stored in memory.
Each counter is updated incrementally in record(), keeping memory O(1)
regardless of how many tasks are run.
"""

from dataclasses import dataclass, field

from workflows.base import WorkflowResult


@dataclass
class WorkflowMetrics:
    """Aggregated metrics for a single workflow across all tasks."""

    workflow_name: str
    total_tasks: int = 0
    successes: int = 0
    failures: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_retries: int = 0
    # Running sums for derived metrics — avoids storing all WorkflowResult objects
    sum_reasoning_steps: int = field(default=0, repr=False)
    # Per-task tool accuracy accumulation: mean of (successful/total) per task,
    # so every task contributes equally regardless of how many tools it uses.
    sum_per_task_tool_accuracy: float = field(default=0.0, repr=False)
    tool_accuracy_tasks: int = field(default=0, repr=False)
    sum_quality_score: float = field(default=0.0, repr=False)
    quality_scored_tasks: int = field(default=0, repr=False)

    # --- Blueprint metrics ---

    @property
    def success_rate(self) -> float:
        """Fraction of tasks completed successfully."""
        return self.successes / self.total_tasks if self.total_tasks else 0.0

    @property
    def failure_rate(self) -> float:
        """Fraction of tasks that failed."""
        return self.failures / self.total_tasks if self.total_tasks else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_tasks if self.total_tasks else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.total_tasks if self.total_tasks else 0.0

    @property
    def avg_retries(self) -> float:
        return self.total_retries / self.total_tasks if self.total_tasks else 0.0

    @property
    def avg_reasoning_steps(self) -> float:
        """Average number of reasoning steps taken across all tasks."""
        return self.sum_reasoning_steps / self.total_tasks if self.total_tasks else 0.0

    @property
    def tool_accuracy(self) -> float:
        """Mean per-task tool accuracy.

        Computed as the average of (successful_calls / total_calls) per task,
        so each task contributes equally regardless of how many tool calls it
        makes.  Tasks with zero tool calls are excluded from the average.
        """
        return (
            self.sum_per_task_tool_accuracy / self.tool_accuracy_tasks
            if self.tool_accuracy_tasks
            else 0.0
        )

    @property
    def avg_quality_score(self) -> float | None:
        """Average LLM-as-judge quality score (0.0–1.0), or None if not evaluated."""
        if self.quality_scored_tasks == 0:
            return None
        return self.sum_quality_score / self.quality_scored_tasks

    def record(self, result: WorkflowResult) -> None:
        """Record a single workflow result and update all counters."""
        self.total_tasks += 1
        if result.success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_tokens += result.total_tokens
        self.total_latency_ms += result.latency_ms
        self.total_retries += result.retries
        self.sum_reasoning_steps += len(result.reasoning_steps)
        if result.tool_calls_total > 0:
            per_task_acc = result.tool_calls_successful / result.tool_calls_total
            self.sum_per_task_tool_accuracy += per_task_acc
            self.tool_accuracy_tasks += 1
        if result.quality_score is not None:
            self.sum_quality_score += result.quality_score
            self.quality_scored_tasks += 1

    def summary(self) -> dict:
        """Return all benchmark metrics as a dict."""
        d = {
            "workflow": self.workflow_name,
            "total_tasks": self.total_tasks,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "tool_accuracy": round(self.tool_accuracy, 4),
            "avg_reasoning_steps": round(self.avg_reasoning_steps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_retries": round(self.avg_retries, 2),
        }
        qs = self.avg_quality_score
        if qs is not None:
            d["avg_quality_score"] = round(qs, 4)
        return d
