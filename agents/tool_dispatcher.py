"""Tool schema definitions and runtime dispatcher for workflow patterns."""

from __future__ import annotations

import json
from typing import Any

from tools import ALL_TOOLS
from tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# OpenAI-compatible tool schemas (used with model.bind_tools())
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "sql_query",
        "description": (
            "Execute a read-only SQL SELECT query against the business SQLite database. "
            "Tables available: customers, orders, products, inventory, payments, revenue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid SQL SELECT statement.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression safely. "
            "Supports: +, -, *, /, //, %, **, sqrt, log, log10, abs, round, min, max, sum, ceil, floor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g. '(100 - 80) / 100').",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "vector_search",
        "description": (
            "Search the business document knowledge base (policies, contracts, rules). "
            "Returns the most relevant document snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "csv_reader",
        "description": (
            "Read rows from a CSV data file. "
            "Valid filenames: customers, orders, products, inventory, payments, revenue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "CSV file name without extension.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return (default 500).",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "python_analysis",
        "description": (
            "Execute Python code for data analysis. "
            "Pre-loaded DataFrames: customers_df, orders_df, products_df, inventory_df, payments_df, revenue_df. "
            "Assign the final result to the variable 'result'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Must assign output to 'result'.",
                }
            },
            "required": ["code"],
        },
    },
]

# Map tool names to BaseTool instances
_TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


class ToolDispatcher:
    """Dispatches tool-call requests to the corresponding BaseTool."""

    def __init__(self, allowed_tools: list[str] | None = None) -> None:
        """
        Args:
            allowed_tools: Restrict dispatch to these tool names.
                           None means all tools are allowed.
        """
        if allowed_tools is not None:
            self._tools = {
                name: _TOOL_MAP[name] for name in allowed_tools if name in _TOOL_MAP
            }
            self._schemas = [s for s in TOOL_SCHEMAS if s["name"] in allowed_tools]
        else:
            self._tools = dict(_TOOL_MAP)
            self._schemas = list(TOOL_SCHEMAS)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """Tool schemas to pass to model.bind_tools()."""
        return self._schemas

    async def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: '{name}'")
        try:
            return await tool.execute(**args)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def result_to_text(self, result: ToolResult) -> str:
        """Serialize a ToolResult to a string for LLM consumption."""
        if not result.success:
            return f"ERROR: {result.error}"
        try:
            return json.dumps(result.data, default=str)
        except (TypeError, ValueError):
            return str(result.data)
