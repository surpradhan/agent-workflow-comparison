"""Workflow 5: Parallel — concurrent tool execution then synthesis.

Pattern:
  Step 1 (Plan)       — LLM identifies multiple independent data queries
  Step 2 (Execute)    — all queries run concurrently with asyncio.gather
  Step 3 (Synthesize) — LLM synthesizes all results into a final answer
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from agents.tool_dispatcher import ToolDispatcher
from tasks.task_registry import Task
from workflows._utils import parse_json
from workflows.base import BaseWorkflow, WorkflowResult

log = logging.getLogger(__name__)

_MAX_QUERIES = 4

_PLAN_SYSTEM = (
    "You are a data analyst. Given a business question, identify ALL independent data queries "
    "needed to answer it comprehensively.\n"
    'Respond with a JSON object: {"queries": [{"tool": "<name>", "args": {...}}, ...]}\n'
    "Available tools: sql_query, calculator, vector_search, csv_reader, python_analysis.\n"
    f"Include at most {_MAX_QUERIES} queries. Focus on queries that can run independently in parallel.\n"
    "Output only valid JSON, no markdown fences."
)


class ParallelWorkflow(BaseWorkflow):
    """Concurrent tool execution followed by LLM synthesis."""

    name = "parallel"
    description = "Runs multiple tool queries concurrently, then synthesizes results"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._dispatcher = ToolDispatcher()

    async def run(self, task: Task) -> WorkflowResult:
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
            # ── Step 1: Plan ───────────────────────────────────────────────
            reasoning.append("Step 1: Planning parallel queries")
            plan_text, tokens = await self._llm.invoke(
                [SystemMessage(content=_PLAN_SYSTEM), HumanMessage(content=task.description)]
            )
            total_tokens += tokens
            retries += self._llm.last_retries
            plan, parse_ok = _parse_plan(plan_text)

            if not parse_ok:
                log.warning("Parallel plan parsing failed for task %s — using fallback query", task.id)
                reasoning.append(
                    "WARNING: Plan parsing failed (LLM returned non-JSON) — using fallback SQL query"
                )
                plan = [{"tool": "sql_query", "args": {"query": "SELECT * FROM revenue LIMIT 20"}}]
            elif not plan:
                reasoning.append("WARNING: LLM returned empty query plan — using fallback SQL query")
                plan = [{"tool": "sql_query", "args": {"query": "SELECT * FROM revenue LIMIT 20"}}]

            if len(plan) > _MAX_QUERIES:
                reasoning.append(
                    f"WARNING: Plan had {len(plan)} queries, capping to {_MAX_QUERIES}"
                )
                plan = plan[:_MAX_QUERIES]

            reasoning.append(f"Planned {len(plan)} parallel queries")
            tool_calls_total += len(plan)
            for item in plan:
                tools_used.append(item.get("tool", "unknown"))

            # ── Step 2: Execute concurrently ───────────────────────────────
            reasoning.append(f"Step 2: Executing {len(plan)} queries in parallel")
            raw_results = await asyncio.gather(
                *[
                    self._dispatcher.dispatch(item.get("tool", ""), item.get("args", {}))
                    for item in plan
                ],
                return_exceptions=True,
            )

            retrieved_parts: list[str] = []
            for item, res in zip(plan, raw_results):
                tool_name = item.get("tool", "unknown")
                if isinstance(res, Exception):
                    retrieved_parts.append(f"[{tool_name}] ERROR: {res}")
                    reasoning.append(f"  {tool_name} raised exception: {res}")
                else:
                    if res.success:
                        tool_calls_successful += 1
                        retrieved_parts.append(
                            f"[{tool_name}]\n{self._dispatcher.result_to_text(res)}"
                        )
                        reasoning.append(f"  {tool_name} completed successfully")
                    else:
                        retrieved_parts.append(f"[{tool_name}] ERROR: {res.error}")
                        reasoning.append(f"  {tool_name} failed: {res.error}")

            combined = "\n\n".join(retrieved_parts)

            # ── Step 3: Synthesize ─────────────────────────────────────────
            reasoning.append("Step 3: Synthesizing results from all parallel queries")
            synth_prompt = (
                f"Question: {task.description}\n\n"
                f"Data from parallel queries:\n{combined}\n\n"
                "Synthesize all the above data into a precise, complete answer."
            )
            answer, tokens = await self._llm.invoke(
                [
                    SystemMessage(content="You are a business analyst. Synthesize data into clear answers."),
                    HumanMessage(content=synth_prompt),
                ]
            )
            total_tokens += tokens
            retries += self._llm.last_retries

            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                answer=answer,
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


def _parse_plan(text: str) -> tuple[list[dict], bool]:
    """Parse the LLM's JSON plan using the shared 3-strategy parser.

    Returns (queries, parse_ok). parse_ok=False means all strategies failed.
    """
    data, ok = parse_json(text)
    return data.get("queries", []), ok
