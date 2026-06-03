"""Integration tests for data layer tools.

These tests exercise the tools with real data structures (in-memory SQLite,
temporary CSV files) rather than mocked interfaces.  No LLM calls are made.

Goals:
- Verify happy-path execution produces correctly shaped ToolResult data.
- Verify error paths (blocked SQL, missing files, bad code) return
  ToolResult(success=False) with a meaningful error message rather than raising.
- Confirm security guards are in place (write-operation blocking, exec sandbox).
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(path: str) -> None:
    """Seed a minimal SQLite database at *path* for SQL tool tests."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE revenue (month TEXT, category TEXT, total_revenue REAL)"
        )
        conn.execute(
            "INSERT INTO revenue VALUES ('2024-03', 'Hardware', 42000.0)"
        )
        conn.execute(
            "INSERT INTO revenue VALUES ('2024-03', 'Software', 31000.0)"
        )
        conn.execute("CREATE TABLE customers (id INTEGER, name TEXT, segment TEXT)")
        conn.execute("INSERT INTO customers VALUES (1, 'Alice', 'Enterprise')")
        conn.execute("INSERT INTO customers VALUES (2, 'Bob', 'SMB')")
        conn.commit()


def _make_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as a CSV file at *path*."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Calculator — no external dependencies
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self) -> None:
        from tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = await tool.execute(expression="(100 - 80) / 100")
        assert result.success is True
        assert result.data["result"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_math_functions(self) -> None:
        from tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = await tool.execute(expression="round(sqrt(144), 2)")
        assert result.success is True
        assert result.data["result"] == pytest.approx(12.0)

    @pytest.mark.asyncio
    async def test_division_by_zero_returns_failure(self) -> None:
        from tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = await tool.execute(expression="1 / 0")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_unsupported_expression_returns_failure(self) -> None:
        """String literals and imports must be rejected."""
        from tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = await tool.execute(expression="'hello' + 'world'")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_power_operator(self) -> None:
        from tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = await tool.execute(expression="2 ** 10")
        assert result.success is True
        assert result.data["result"] == 1024

    @pytest.mark.asyncio
    async def test_negative_unary(self) -> None:
        from tools.calculator import CalculatorTool

        tool = CalculatorTool()
        result = await tool.execute(expression="-abs(-5)")
        assert result.success is True
        assert result.data["result"] == -5


# ---------------------------------------------------------------------------
# SQLQueryTool — uses in-memory/temp SQLite, no prod DB required
# ---------------------------------------------------------------------------


class TestSQLQueryTool:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        path = str(tmp_path / "test.db")
        _make_db(path)
        return path

    @pytest.mark.asyncio
    async def test_select_returns_rows(self, db_path: str) -> None:
        from tools.sql_query import SQLQueryTool

        tool = SQLQueryTool(db_path=db_path)
        result = await tool.execute(query="SELECT * FROM revenue")
        assert result.success is True
        assert result.data["row_count"] == 2
        assert "month" in result.data["columns"]

    @pytest.mark.asyncio
    async def test_aggregation_query(self, db_path: str) -> None:
        from tools.sql_query import SQLQueryTool

        tool = SQLQueryTool(db_path=db_path)
        result = await tool.execute(
            query="SELECT SUM(total_revenue) AS total FROM revenue"
        )
        assert result.success is True
        assert result.data["rows"][0]["total"] == pytest.approx(73000.0)

    @pytest.mark.asyncio
    async def test_filtered_query(self, db_path: str) -> None:
        from tools.sql_query import SQLQueryTool

        tool = SQLQueryTool(db_path=db_path)
        result = await tool.execute(
            query="SELECT * FROM customers WHERE segment = 'Enterprise'"
        )
        assert result.success is True
        assert result.data["row_count"] == 1
        assert result.data["rows"][0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_drop_table_is_blocked(self, db_path: str) -> None:
        from tools.sql_query import SQLQueryTool

        tool = SQLQueryTool(db_path=db_path)
        result = await tool.execute(query="DROP TABLE revenue")
        assert result.success is False
        assert result.error is not None
        assert "DROP" in result.error

    @pytest.mark.asyncio
    async def test_delete_is_blocked(self, db_path: str) -> None:
        from tools.sql_query import SQLQueryTool

        tool = SQLQueryTool(db_path=db_path)
        result = await tool.execute(query="DELETE FROM customers WHERE id = 1")
        assert result.success is False
        assert result.error is not None
        assert "DELETE" in result.error

    @pytest.mark.asyncio
    async def test_invalid_sql_returns_failure(self, db_path: str) -> None:
        from tools.sql_query import SQLQueryTool

        tool = SQLQueryTool(db_path=db_path)
        result = await tool.execute(query="SELCT * FORM revenue")  # typos
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_nonexistent_table_returns_failure(self, db_path: str) -> None:
        from tools.sql_query import SQLQueryTool

        tool = SQLQueryTool(db_path=db_path)
        result = await tool.execute(query="SELECT * FROM nonexistent_table")
        assert result.success is False


# ---------------------------------------------------------------------------
# CSVReaderTool — uses a temporary directory with a real CSV file
# ---------------------------------------------------------------------------


class TestCSVReaderTool:
    @pytest.fixture
    def csv_dir(self, tmp_path: Path, monkeypatch) -> Path:
        """Create a temp dir with a minimal customers.csv and patch settings."""
        _make_csv(
            tmp_path / "customers.csv",
            [
                {"id": "1", "name": "Alice", "segment": "Enterprise"},
                {"id": "2", "name": "Bob", "segment": "SMB"},
                {"id": "3", "name": "Carol", "segment": "Enterprise"},
            ],
        )
        from config import settings as _settings
        monkeypatch.setattr(_settings, "csv_dir", tmp_path)
        return tmp_path

    @pytest.mark.asyncio
    async def test_reads_csv_rows(self, csv_dir: Path) -> None:
        from tools.csv_reader import CSVReaderTool

        tool = CSVReaderTool()
        result = await tool.execute(filename="customers")
        assert result.success is True
        assert result.data["row_count"] == 3
        assert "id" in result.data["columns"]

    @pytest.mark.asyncio
    async def test_limit_is_respected(self, csv_dir: Path) -> None:
        from tools.csv_reader import CSVReaderTool

        tool = CSVReaderTool()
        result = await tool.execute(filename="customers", limit=2)
        assert result.success is True
        assert result.data["row_count"] == 2
        assert result.data["truncated"] is True

    @pytest.mark.asyncio
    async def test_unknown_filename_returns_failure(self, csv_dir: Path) -> None:
        from tools.csv_reader import CSVReaderTool

        tool = CSVReaderTool()
        result = await tool.execute(filename="nonexistent")
        assert result.success is False
        assert result.error is not None
        assert "nonexistent" in result.error

    @pytest.mark.asyncio
    async def test_missing_file_returns_failure(self, csv_dir: Path) -> None:
        """File is in VALID_FILES but doesn't exist on disk."""
        from tools.csv_reader import CSVReaderTool

        tool = CSVReaderTool()
        # "orders" is valid but we didn't create orders.csv in the fixture
        result = await tool.execute(filename="orders")
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_accepts_csv_extension_suffix(self, csv_dir: Path) -> None:
        """Tool should strip .csv suffix from filename argument."""
        from tools.csv_reader import CSVReaderTool

        tool = CSVReaderTool()
        result = await tool.execute(filename="customers.csv")
        assert result.success is True


# ---------------------------------------------------------------------------
# PythonAnalysisTool — exercises exec sandbox with injected DataFrames
# ---------------------------------------------------------------------------


class TestPythonAnalysisTool:
    @pytest.fixture
    def csv_dir(self, tmp_path: Path, monkeypatch) -> Path:
        """Create minimal CSVs and patch settings.csv_dir."""
        for name in ("customers", "orders", "products", "inventory", "payments", "revenue"):
            _make_csv(tmp_path / f"{name}.csv", [{"col": "val"}])
        from config import settings as _settings
        monkeypatch.setattr(_settings, "csv_dir", tmp_path)
        return tmp_path

    @pytest.mark.asyncio
    async def test_simple_computation(self, csv_dir: Path) -> None:
        from tools.python_analysis import PythonAnalysisTool

        tool = PythonAnalysisTool()
        result = await tool.execute(code="result = 2 + 2")
        assert result.success is True
        assert result.data["result"] == 4

    @pytest.mark.asyncio
    async def test_result_variable_required(self, csv_dir: Path) -> None:
        """Code that runs but never sets 'result' still succeeds with result=None."""
        from tools.python_analysis import PythonAnalysisTool

        tool = PythonAnalysisTool()
        result = await tool.execute(code="x = 42  # no 'result' variable")
        assert result.success is True
        assert result.data["result"] is None

    @pytest.mark.asyncio
    async def test_syntax_error_returns_failure(self, csv_dir: Path) -> None:
        from tools.python_analysis import PythonAnalysisTool

        tool = PythonAnalysisTool()
        result = await tool.execute(code="def broken(: pass")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_runtime_error_returns_failure(self, csv_dir: Path) -> None:
        from tools.python_analysis import PythonAnalysisTool

        tool = PythonAnalysisTool()
        result = await tool.execute(code="result = 1 / 0")
        assert result.success is False
        assert result.error is not None
        assert "division" in result.error.lower() or "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_filesystem_access_is_blocked(self, csv_dir: Path) -> None:
        """__builtins__ restriction must prevent open() calls."""
        from tools.python_analysis import PythonAnalysisTool

        tool = PythonAnalysisTool()
        result = await tool.execute(code="open('/etc/passwd')")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_import_is_blocked(self, csv_dir: Path) -> None:
        """__import__ must not be available in exec namespace."""
        from tools.python_analysis import PythonAnalysisTool

        tool = PythonAnalysisTool()
        result = await tool.execute(code="import os; result = os.getcwd()")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_dataframe_available(self, csv_dir: Path) -> None:
        """customers_df and other pre-loaded DataFrames must be accessible."""
        from tools.python_analysis import PythonAnalysisTool

        tool = PythonAnalysisTool()
        result = await tool.execute(code="result = len(customers_df)")
        assert result.success is True
        assert isinstance(result.data["result"], int)
