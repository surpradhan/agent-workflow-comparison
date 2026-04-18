"""Base workflow interface that all 10 workflow patterns implement."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tasks.task_registry import Task


class WorkflowResult(BaseModel):
    """Standardized result from a workflow execution.

    Metric definitions (absolute, non-overlapping):
      success        — True iff the workflow produced a non-empty answer.
                       Measures *completion*, not answer quality or tool use.
      tool_accuracy  — fraction of tool calls that returned success=True.
                       Measures *execution quality* of the tool layer only.
      quality_score  — LLM-as-judge score 0.0–1.0 against ground truth.
                       Measures *answer correctness*, set post-construction.
    """

    model_config = ConfigDict(frozen=False)

    @field_validator("tools_used", mode="before")
    @classmethod
    def _drop_none_tools(cls, v: list) -> list:
        return [t for t in (v or []) if t is not None]

    @model_validator(mode="after")
    def _derive_success(self) -> "WorkflowResult":
        """Success = workflow produced a non-empty answer.

        Overrides any value set by the workflow so the definition is
        consistent across all 10 patterns.
        """
        self.success = bool(self.answer and self.answer.strip())
        return self

    task_id: str
    workflow_name: str
    answer: str | None = None
    success: bool = False            # derived: bool(answer) — do not set manually
    reasoning_steps: list[str] = []
    tools_used: list[str] = []
    tool_calls_total: int = 0        # total tool invocations attempted
    tool_calls_successful: int = 0   # invocations that returned success=True
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
