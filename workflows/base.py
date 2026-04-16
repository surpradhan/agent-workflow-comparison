"""Base workflow interface that all 10 workflow patterns implement."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from tasks.task_registry import Task


class WorkflowResult(BaseModel):
    """Standardized result from a workflow execution.

    Carries all data needed to compute the 7 benchmark metrics:
    success rate, failure rate, tool accuracy, reasoning steps,
    latency, token cost, retries.
    """

    task_id: str
    workflow_name: str
    answer: Any = None
    success: bool = False
    reasoning_steps: list[str] = []
    tools_used: list[str] = []
    tool_calls_total: int = 0       # total tool invocations made
    tool_calls_successful: int = 0  # invocations that returned success=True
    total_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    error: str | None = None


class BaseWorkflow(ABC):
    """Abstract base class for all workflow patterns."""

    name: str
    description: str

    @abstractmethod
    async def run(self, task: Task) -> WorkflowResult:
        """Execute the workflow on a given task."""
        ...
