# Import definitions to trigger registration
import tasks.definitions  # noqa: F401
from tasks.task_registry import Task, TaskLevel, TaskRegistry, registry

__all__ = ["Task", "TaskLevel", "TaskRegistry", "registry"]
