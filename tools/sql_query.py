"""SQL query tool for executing queries against the SQLite benchmark database."""

import sqlite3
from typing import Any

from config import settings
from tools.base import BaseTool, ToolResult

# Disallowed SQL operations for safety
BLOCKED_KEYWORDS = {"DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"}


class SQLQueryTool(BaseTool):
    name = "sql_query"
    description = (
        "Execute a read-only SQL query against the business database. "
        "Available tables: customers, orders, products, inventory, payments, revenue."
    )

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.database_url.replace("sqlite:///", "")

    def _validate_query(self, query: str) -> str | None:
        """Return an error message if the query is unsafe, else None."""
        upper = query.upper().strip()
        for keyword in BLOCKED_KEYWORDS:
            if keyword in upper.split():
                return f"Write operation '{keyword}' is not allowed. Only SELECT queries are permitted."
        return None

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        error = self._validate_query(query)
        if error:
            return ToolResult(success=False, error=error)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = [dict(row) for row in cursor.fetchall()]
            return ToolResult(success=True, data={"columns": columns, "rows": rows, "row_count": len(rows)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
