"""Task definitions and registry for benchmark evaluation."""

from enum import IntEnum
from typing import Any

from pydantic import BaseModel


class TaskLevel(IntEnum):
    """Task difficulty levels."""

    RETRIEVAL = 1       # Simple lookups (revenue, product search)
    ANALYTICAL = 2      # Trends, segmentation, aggregation
    REASONING = 3       # Multi-step reasoning, performance explanation
    DECISION = 4        # Strategy recommendations, trade-off analysis


class Task(BaseModel):
    """A single benchmark task."""

    id: str
    description: str
    level: TaskLevel
    expected_tools: list[str] = []
    ground_truth: Any = None
    evaluation_criteria: str = ""


class TaskRegistry:
    """Central registry of all benchmark tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def register(self, task: Task) -> None:
        self._tasks[task.id] = task

    def get(self, task_id: str) -> Task:
        if task_id not in self._tasks:
            raise ValueError(f"Task '{task_id}' not found. Available: {sorted(self._tasks)}")
        return self._tasks[task_id]

    def get_by_level(self, level: TaskLevel) -> list[Task]:
        return [t for t in self._tasks.values() if t.level == level]

    def all(self) -> list[Task]:
        return list(self._tasks.values())


registry = TaskRegistry()
