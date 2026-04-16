"""CSV reader tool for direct file-based data access."""

import csv
from typing import Any

from config import settings
from tools.base import BaseTool, ToolResult


class CSVReaderTool(BaseTool):
    name = "csv_reader"
    description = (
        "Read data from CSV files. Available files: customers.csv, orders.csv, "
        "products.csv, inventory.csv, payments.csv, revenue.csv."
    )

    VALID_FILES = {"customers", "orders", "products", "inventory", "payments", "revenue"}

    async def execute(self, filename: str, limit: int = 500, **kwargs: Any) -> ToolResult:
        stem = filename.replace(".csv", "")
        if stem not in self.VALID_FILES:
            return ToolResult(
                success=False,
                error=f"Unknown file '{filename}'. Valid files: {', '.join(sorted(self.VALID_FILES))}",
            )

        path = settings.csv_dir / f"{stem}.csv"
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    rows.append(row)

            return ToolResult(
                success=True,
                data={
                    "filename": f"{stem}.csv",
                    "columns": list(rows[0].keys()) if rows else [],
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": len(rows) == limit,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
