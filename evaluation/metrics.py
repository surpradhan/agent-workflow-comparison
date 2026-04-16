"""Evaluation metrics for comparing workflow performance.

Tracks all 7 metrics from the benchmark blueprint:
task success rate, tool accuracy, reasoning steps, latency, token cost,
retries, and failure rate.
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
    # Per-result tracking for derived metrics
    results: list[WorkflowResult] = field(default_factory=list)

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
        if not self.results:
            return 0.0
        return sum(len(r.reasoning_steps) for r in self.results) / len(self.results)

    @property
    def tool_accuracy(self) -> float:
        """Fraction of tool calls that returned success=True.

        Requires WorkflowResult.tool_call_outcomes to be populated by the workflow.
        Falls back to 0.0 if no outcomes were recorded.
        """
        total_calls = sum(r.tool_calls_total for r in self.results)
        successful_calls = sum(r.tool_calls_successful for r in self.results)
        return successful_calls / total_calls if total_calls else 0.0

    def record(self, result: WorkflowResult) -> None:
        """Record a single workflow result and update all counters."""
        self.results.append(result)
        self.total_tasks += 1
        if result.success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_tokens += result.total_tokens
        self.total_latency_ms += result.latency_ms
        self.total_retries += result.retries

    def summary(self) -> dict:
        """Return all 7 blueprint metrics as a dict."""
        return {
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
