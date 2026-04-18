"""Base workflow interface that all 10 workflow patterns implement."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, field_validator

from tasks.task_registry import Task


class WorkflowResult(BaseModel):
    """Standardized result from a workflow execution.

    Carries all data needed to compute the 7 benchmark metrics:
    success rate, failure rate, tool accuracy, reasoning steps,
    latency, token cost, retries.

    Not frozen: quality_score is populated by the judge phase after construction.
    """

    model_config = ConfigDict(frozen=False)

    @field_validator("tools_used", mode="before")
    @classmethod
    def _drop_none_tools(cls, v: list) -> list:
        return [t for t in (v or []) if t is not None]

    task_id: str
    workflow_name: str
    answer: str | None = None
    success: bool = False
    reasoning_steps: list[str] = []
    tools_used: list[str] = []
    tool_calls_total: int = 0       # total tool invocations made
    tool_calls_successful: int = 0  # invocations that returned success=True
    total_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    error: str | None = None
    quality_score: float | None = None  # 0.0–1.0 LLM-as-judge score; None if not evaluated


class BaseWorkflow(ABC):
    """Abstract base class for all workflow patterns."""

    name: str
    description: str

    def _validate_task(self, task: Task) -> str | None:
        """Return an error string if the task is unusable, else None."""
        if not task.description or not task.description.strip():
            return "Task description is empty"
        return None

    @abstractmethod
    async def run(self, task: Task) -> WorkflowResult:
        """Execute the workflow on a given task."""
        ...
