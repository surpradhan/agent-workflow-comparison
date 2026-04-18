"""Workflow 4: Routing — classify task then dispatch to a specialized handler.

The router LLM classifies the task into one of three tracks:
  - "retrieval"  → single SQL lookup, direct answer
  - "analytical" → SQL + Python analysis, structured computation
  - "complex"    → full ReAct agent with all tools

Each handler is a compact, purpose-built sub-workflow.
"""

from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from agents.tool_dispatcher import ToolDispatcher
from tasks.task_registry import Task
from workflows.base import BaseWorkflow, WorkflowResult

_ROUTE_LABELS = ("retrieval", "analytical", "complex")

_ROUTER_SYSTEM = (
    "You are a task router. Classify the following business question into exactly ONE category:\n"
    "- 'retrieval': a simple lookup — find a specific value, list items, or retrieve a fact.\n"
    "- 'analytical': requires aggregation, trend analysis, or metric computation.\n"
    "- 'complex': requires multi-step reasoning, cross-referencing multiple data sources, "
    "or strategic decision-making.\n\n"
    "Respond with exactly one word: retrieval, analytical, or complex."
)

# Handler tuple structure: (answer, handler_success, reasoning, tools, tc_total, tc_ok, tokens, retries)
_HandlerResult = tuple[str | None, bool, list[str], list[str], int, int, int, int]


class RoutingWorkflow(BaseWorkflow):
    """Router classifies task, then dispatches to a specialized handler."""

    name = "routing"
    description = "Router classifies task, dispatches to a specialized handler"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._dispatcher = ToolDispatcher()

    async def run(self, task: Task) -> WorkflowResult:
        # Fix 11: validate task
        if err := self._validate_task(task):
            return WorkflowResult(
                task_id=task.id, workflow_name=self.name, success=False, error=err
            )

        start = time.perf_counter()
        reasoning: list[str] = []
        tools_used: list[str] = []
        total_tokens = 0
        tool_calls_total = 0
        tool_calls_successful = 0
        retries = 0

        try:
            # ── Route ──────────────────────────────────────────────────────
            route_text, tokens = await self._llm.invoke(
                [SystemMessage(content=_ROUTER_SYSTEM), HumanMessage(content=task.description)]
            )
            total_tokens += tokens
            retries += self._llm.last_retries
            route = _parse_route(route_text)
            reasoning.append(f"Router classified task as: '{route}'")

            # ── Dispatch ───────────────────────────────────────────────────
            if route == "retrieval":
                answer, handler_ok, step_steps, step_tools, tc_total, tc_ok, step_tokens, step_retries = (
                    await self._handle_retrieval(task)
                )
            elif route == "analytical":
                answer, handler_ok, step_steps, step_tools, tc_total, tc_ok, step_tokens, step_retries = (
                    await self._handle_analytical(task)
                )
            else:
                answer, handler_ok, step_steps, step_tools, tc_total, tc_ok, step_tokens, step_retries = (
                    await self._handle_complex(task)
                )

            reasoning.extend(step_steps)
            tools_used.extend(step_tools)
            tool_calls_total += tc_total
            tool_calls_successful += tc_ok
            total_tokens += step_tokens
            retries += step_retries

            # Fix 10: propagate handler success instead of always True
            # Also apply Fix 3: all-tool-failure → success=False
            success = handler_ok and (tool_calls_total == 0 or tool_calls_successful > 0)

            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                answer=answer,
                success=success,
                reasoning_steps=reasoning,
                tools_used=list(dict.fromkeys(tools_used)),
                tool_calls_total=tool_calls_total,
                tool_calls_successful=tool_calls_successful,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                retries=retries,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                success=False,
                reasoning_steps=reasoning,
                tools_used=list(dict.fromkeys(tools_used)),
                tool_calls_total=tool_calls_total,
                tool_calls_successful=tool_calls_successful,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                retries=retries,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Handlers — each returns _HandlerResult
    # ------------------------------------------------------------------

    async def _handle_retrieval(self, task: Task) -> _HandlerResult:
        """Simple retrieval: generate SQL, run it, format answer."""
        reasoning = ["Handler: retrieval — generating lookup query"]
        tools_used: list[str] = []
        tc_total = tc_ok = tokens_total = handler_retries = 0

        sql_text, tokens = await self._llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a SQL expert. Write a single SQL SELECT query to answer the question. "
                        "Tables: customers, orders, products, inventory, payments, revenue. "
                        "Output ONLY the raw SQL query, no explanation or markdown."
                    )
                ),
                HumanMessage(content=task.description),
            ]
        )
        tokens_total += tokens
        handler_retries += self._llm.last_retries
        query = _strip_code_fence(sql_text)

        result = await self._dispatcher.dispatch("sql_query", {"query": query})
        tc_total += 1
        tools_used.append("sql_query")
        if result.success:
            tc_ok += 1
        data_text = self._dispatcher.result_to_text(result)
        reasoning.append(f"Executed SQL lookup, result: {'ok' if result.success else 'failed'}")

        answer_text, tokens = await self._llm.invoke(
            [
                SystemMessage(content="You are a business analyst. Answer the question using the data provided."),
                HumanMessage(content=f"Question: {task.description}\n\nData:\n{data_text}"),
            ]
        )
        tokens_total += tokens
        handler_retries += self._llm.last_retries
        reasoning.append("Formatted final answer from query result")
        handler_ok = tc_ok > 0  # Fix 10: handler reports its own success
        return answer_text, handler_ok, reasoning, tools_used, tc_total, tc_ok, tokens_total, handler_retries

    async def _handle_analytical(self, task: Task) -> _HandlerResult:
        """Analytical: SQL retrieval + Python analysis."""
        reasoning = ["Handler: analytical — SQL retrieval then Python analysis"]
        tools_used: list[str] = []
        tc_total = tc_ok = tokens_total = handler_retries = 0

        # SQL retrieval
        sql_text, tokens = await self._llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a SQL expert. Write a SQL SELECT query to retrieve all data needed "
                        "for this analysis. Tables: customers, orders, products, inventory, payments, revenue. "
                        "Output ONLY the raw SQL query."
                    )
                ),
                HumanMessage(content=task.description),
            ]
        )
        tokens_total += tokens
        handler_retries += self._llm.last_retries
        query = _strip_code_fence(sql_text)

        sql_result = await self._dispatcher.dispatch("sql_query", {"query": query})
        tc_total += 1
        tools_used.append("sql_query")
        if sql_result.success:
            tc_ok += 1
        data_text = self._dispatcher.result_to_text(sql_result)
        reasoning.append(f"Retrieved data via SQL: {'ok' if sql_result.success else 'failed'}")

        # Python analysis
        code_prompt = (
            f"Question: {task.description}\n\nRaw data (JSON):\n{data_text}\n\n"
            "Write Python code to analyze this data using pandas or plain Python. "
            "Parse the JSON, compute the required metrics, and assign the final result "
            "(a string or dict) to a variable called 'result'. Output ONLY the code."
        )
        code_text, tokens = await self._llm.invoke(
            [
                SystemMessage(content="You are a Python data analyst. Output only code, no markdown."),
                HumanMessage(content=code_prompt),
            ]
        )
        tokens_total += tokens
        handler_retries += self._llm.last_retries
        code = _strip_code_fence(code_text)

        py_result = await self._dispatcher.dispatch("python_analysis", {"code": code})
        tc_total += 1
        tools_used.append("python_analysis")
        if py_result.success:
            tc_ok += 1
        analysis_text = self._dispatcher.result_to_text(py_result)
        reasoning.append(f"Python analysis: {'ok' if py_result.success else 'failed'}")

        answer_text, tokens = await self._llm.invoke(
            [
                SystemMessage(content="You are a business analyst. Provide a clear, precise answer."),
                HumanMessage(
                    content=(
                        f"Question: {task.description}\n\n"
                        f"Analysis result:\n{analysis_text}\n\n"
                        "Provide a precise, concise answer."
                    )
                ),
            ]
        )
        tokens_total += tokens
        handler_retries += self._llm.last_retries
        reasoning.append("Synthesized analytical answer")
        handler_ok = tc_ok > 0  # Fix 10
        return answer_text, handler_ok, reasoning, tools_used, tc_total, tc_ok, tokens_total, handler_retries

    async def _handle_complex(self, task: Task) -> _HandlerResult:
        """Complex: delegate to a fresh tool-using ReAct agent instance."""
        from workflows.tool_using import ToolUsingWorkflow
        reasoning = ["Handler: complex — delegating to full tool-using agent"]
        inner = await ToolUsingWorkflow().run(task)
        # Fix 10: propagate inner workflow's success flag
        return (
            inner.answer,
            inner.success,
            reasoning + inner.reasoning_steps,
            inner.tools_used,
            inner.tool_calls_total,
            inner.tool_calls_successful,
            inner.total_tokens,
            inner.retries,
        )


def _parse_route(text: str) -> str:
    """Extract a valid route label from the LLM response.

    Strategy:
    1. Try exact first-token match (LLM was asked for exactly one word).
    2. Fall back to word-boundary regex search, checking more-specific
       labels before generic ones to avoid false substring matches
       (e.g. "analytical retrieval" should return "analytical", not "retrieval").
    """
    normalized = text.strip().lower()
    # Primary: first non-whitespace token must be exactly a label
    first_token = re.split(r"\W+", normalized)[0] if normalized else ""
    if first_token in _ROUTE_LABELS:
        return first_token
    # Secondary: word-boundary search — more-specific labels checked first
    for label in ("analytical", "retrieval", "complex"):
        if re.search(rf"\b{label}\b", normalized):
            return label
    return "complex"  # safe default


def _strip_code_fence(text: str) -> str:
    """Remove leading/trailing markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner).strip()
    return text
