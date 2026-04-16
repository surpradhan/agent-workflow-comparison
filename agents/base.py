"""Base agent interface used by workflow patterns that delegate to agents."""

from abc import ABC, abstractmethod
from typing import Any

from tools.base import BaseTool


class BaseAgent(ABC):
    """Abstract base class for agents used within workflow patterns."""

    name: str
    description: str
    tools: list[BaseTool]

    @abstractmethod
    async def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the agent and return its response."""
        ...
