"""Tests for the shared tool implementations."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.calculator import CalculatorTool
from tools.csv_reader import CSVReaderTool
from tools.python_analysis import PythonAnalysisTool
from tools.sql_query import SQLQueryTool
from tools.vector_search import VectorSearchTool

# --- SQL Query Tool ---

class TestSQLQueryTool:
    def setup_method(self):
        self.tool = SQLQueryTool()

    @pytest.mark.asyncio
    async def test_select_query(self):
        result = await self.tool.execute(query="SELECT COUNT(*) as cnt FROM customers")
        assert result.success
        assert result.data["rows"][0]["cnt"] == 200

    @pytest.mark.asyncio
    async def test_query_with_filter(self):
        result = await self.tool.execute(
            query="SELECT COUNT(*) as cnt FROM products WHERE category='Software'"
        )
        assert result.success
        assert result.data["rows"][0]["cnt"] == 10

    @pytest.mark.asyncio
    async def test_blocks_write_operations(self):
        result = await self.tool.execute(query="DROP TABLE customers")
        assert not result.success
        assert "not allowed" in result.error

    @pytest.mark.asyncio
    async def test_invalid_sql(self):
        result = await self.tool.execute(query="SELECT * FROM nonexistent_table")
        assert not result.success


# --- CSV Reader Tool ---

class TestCSVReaderTool:
    def setup_method(self):
        self.tool = CSVReaderTool()

    @pytest.mark.asyncio
    async def test_read_customers(self):
        result = await self.tool.execute(filename="customers.csv", limit=5)
        assert result.success
        assert result.data["row_count"] == 5
        assert "customer_id" in result.data["columns"]

    @pytest.mark.asyncio
    async def test_read_with_csv_extension(self):
        result = await self.tool.execute(filename="products", limit=3)
        assert result.success
        assert result.data["row_count"] == 3

    @pytest.mark.asyncio
    async def test_invalid_filename(self):
        result = await self.tool.execute(filename="nonexistent.csv")
        assert not result.success
        assert "Unknown file" in result.error


# --- Vector Search Tool ---

class TestVectorSearchTool:
    def setup_method(self):
        self.tool = VectorSearchTool()

    @pytest.mark.asyncio
    async def test_search_discount_policy(self):
        result = await self.tool.execute(query="discount policy for enterprise customers")
        assert result.success
        assert len(result.data["results"]) > 0
        # Should find business rules about discounts
        texts = " ".join(r["text"] for r in result.data["results"])
        assert "discount" in texts.lower()

    @pytest.mark.asyncio
    async def test_search_kpi(self):
        result = await self.tool.execute(query="how is gross margin calculated")
        assert result.success
        assert len(result.data["results"]) > 0


# --- Calculator Tool ---

class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        result = await self.tool.execute(expression="(100 - 30) / 100 * 100")
        assert result.success
        assert result.data["result"] == 70.0

    @pytest.mark.asyncio
    async def test_functions(self):
        result = await self.tool.execute(expression="round(sqrt(144), 2)")
        assert result.success
        assert result.data["result"] == 12.0

    @pytest.mark.asyncio
    async def test_rejects_unsafe_code(self):
        result = await self.tool.execute(expression="__import__('os').system('ls')")
        assert not result.success


# --- Python Analysis Tool ---

class TestPythonAnalysisTool:
    def setup_method(self):
        self.tool = PythonAnalysisTool()

    @pytest.mark.asyncio
    async def test_basic_analysis(self):
        result = await self.tool.execute(code="result = len(customers_df)")
        assert result.success
        assert result.data["result"] == 200

    @pytest.mark.asyncio
    async def test_pandas_operations(self):
        result = await self.tool.execute(
            code="result = orders_df.groupby('status')['total_amount'].sum().to_dict()"
        )
        assert result.success
        assert "delivered" in result.data["result"]

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        result = await self.tool.execute(code="result = [[[")
        assert not result.success
