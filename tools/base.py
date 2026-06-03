"""Base tool interface that all benchmark tools implement."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standardized result returned by every tool invocation."""

    success: bool
    data: Any = None
    error: str | None = None


class BaseTool(ABC):
    """Abstract base class for benchmark tools."""

    name: str
    description: str

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Subclasses may add explicit parameters before **kwargs for type safety.
        E.g.: async def execute(self, query: str, **kwargs: Any) -> ToolResult
        """
        ...
