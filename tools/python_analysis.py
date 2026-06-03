"""Python analysis tool for executing data analysis code.

NOTE: exec() is used here with a restricted __builtins__ to block filesystem,
network, and subprocess access. Allowed builtins cover only data analysis needs.
This is a benchmark tool running in a controlled local environment — not a
general-purpose sandbox. Do not expose this tool to untrusted user input.
"""

import contextlib
import io
from typing import Any

import pandas as pd

from config import settings
from tools.base import BaseTool, ToolResult

# Builtins available to executed analysis code.
# Excludes: open, __import__, exec, eval, compile, input, and all OS/network access.
_ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "hasattr": hasattr, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "print": print, "range": range, "repr": repr, "reversed": reversed,
    "round": round, "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}


class PythonAnalysisTool(BaseTool):
    name = "python_analysis"
    description = (
        "Execute Python code for data analysis. Has access to pandas (as pd) and "
        "pre-loaded DataFrames: customers_df, orders_df, products_df, inventory_df, "
        "payments_df, revenue_df. Assign your final answer to a variable named 'result'."
    )

    def __init__(self) -> None:
        self._dataframes: dict[str, pd.DataFrame] | None = None

    def _load_dataframes(self) -> dict[str, pd.DataFrame]:
        if self._dataframes is None:
            csv_dir = settings.csv_dir
            self._dataframes = {
                "customers_df": pd.read_csv(csv_dir / "customers.csv"),
                "orders_df": pd.read_csv(csv_dir / "orders.csv"),
                "products_df": pd.read_csv(csv_dir / "products.csv"),
                "inventory_df": pd.read_csv(csv_dir / "inventory.csv"),
                "payments_df": pd.read_csv(csv_dir / "payments.csv"),
                "revenue_df": pd.read_csv(csv_dir / "revenue.csv"),
            }
        return self._dataframes

    async def execute(self, code: str, **kwargs: Any) -> ToolResult:
        dfs = self._load_dataframes()

        namespace: dict[str, Any] = {
            "__builtins__": _ALLOWED_BUILTINS,
            "pd": pd,
            **dfs,
        }

        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code, "<analysis>", "exec"), namespace)

            result = namespace.get("result", None)
            stdout_val = stdout_capture.getvalue()

            if isinstance(result, pd.DataFrame):
                result = result.to_dict(orient="records")
            elif isinstance(result, pd.Series):
                result = result.to_dict()

            return ToolResult(
                success=True,
                data={
                    "result": result,
                    "stdout": stdout_val if stdout_val else None,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Execution error: {e}")
