"""Workflow 3: Chain — sequential multi-step pipeline.

Each step's output feeds directly into the next:
  Step 1 (Decompose)  — understand the task, produce a JSON retrieval plan
  Step 2 (Retrieve)   — execute the plan (tool calls)
  Step 3 (Analyze)    — interpret the retrieved data
  Step 4 (Synthesize) — produce the final answer
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from agents.tool_dispatcher import ToolDispatcher
from tasks.task_registry import Task
from workflows.base import BaseWorkflow, WorkflowResult

log = logging.getLogger(__name__)

_MAX_QUERIES = 3


class ChainWorkflow(BaseWorkflow):
    """Linear four-step pipeline: decompose → retrieve → analyze → synthesize."""

    name = "chain"
    description = "Sequential multi-step pipeline (4 stages)"

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
            # ── Step 1: Decompose ──────────────────────────────────────────
            reasoning.append("Step 1: Decomposing task into a data retrieval plan")
            decompose_prompt = (
                f"Task: {task.description}\n\n"
                "Identify what data you need to answer this question. "
                'Respond with a JSON object: {"queries": [{"tool": "<name>", "args": {...}}, ...]}\n'
                "Use only these tools: sql_query, calculator, vector_search, csv_reader, python_analysis. "
                f"Include at most {_MAX_QUERIES} queries. Be specific with SQL and args."
            )
            plan_text, tokens = await self._llm.invoke(
                [
                    SystemMessage(content="You are a data analyst. Output only valid JSON, no markdown."),
                    HumanMessage(content=decompose_prompt),
                ]
            )
            total_tokens += tokens
            retries += self._llm.last_retries
            plan, parse_ok = _parse_plan(plan_text)

            # Fix 1: surface parse failures in reasoning instead of silently degrading
            if not parse_ok:
                log.warning("Chain plan parsing failed for task %s — LLM returned non-JSON", task.id)
                reasoning.append(
                    "WARNING: Plan parsing failed (LLM returned non-JSON) — "
                    "proceeding with no retrieval data"
                )
            elif not plan:
                reasoning.append("WARNING: LLM returned an empty query plan — skipping retrieval")

            # ── Step 2: Retrieve ───────────────────────────────────────────
            reasoning.append(f"Step 2: Executing {len(plan)} retrieval queries")
            retrieved_parts: list[str] = []
            for item in plan:
                tool_name = item.get("tool", "")
                args = item.get("args", {})
                tools_used.append(tool_name)
                tool_calls_total += 1
                result = await self._dispatcher.dispatch(tool_name, args)
                if result.success:
                    tool_calls_successful += 1
                    retrieved_parts.append(
                        f"[{tool_name}]\n{self._dispatcher.result_to_text(result)}"
                    )
                else:
                    retrieved_parts.append(f"[{tool_name}] ERROR: {result.error}")
                reasoning.append(f"  Retrieved via {tool_name}: {'ok' if result.success else 'failed'}")
            retrieved_data = "\n\n".join(retrieved_parts) if retrieved_parts else "(no data retrieved)"

            # Fix 6: short-circuit if tools were called but every one failed —
            # proceeding to analyze/synthesize would produce confident hallucination.
            if tool_calls_total > 0 and tool_calls_successful == 0:
                latency_ms = (time.perf_counter() - start) * 1000
                return WorkflowResult(
                    task_id=task.id,
                    workflow_name=self.name,
                    success=False,
                    reasoning_steps=reasoning,
                    tools_used=list(dict.fromkeys(tools_used)),
                    tool_calls_total=tool_calls_total,
                    tool_calls_successful=0,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error="All retrieval queries failed — cannot synthesize answer",
                )

            # ── Step 3: Analyze ────────────────────────────────────────────
            reasoning.append("Step 3: Analyzing retrieved data")
            analyze_prompt = (
                f"Task: {task.description}\n\n"
                f"Retrieved data:\n{retrieved_data}\n\n"
                "Analyze this data carefully. Identify key patterns, numbers, and insights "
                "needed to answer the task. Be thorough but concise."
            )
            analysis, tokens = await self._llm.invoke(
                [
                    SystemMessage(content="You are a business analyst. Analyze data precisely."),
                    HumanMessage(content=analyze_prompt),
                ]
            )
            total_tokens += tokens
            retries += self._llm.last_retries

            # ── Step 4: Synthesize ─────────────────────────────────────────
            reasoning.append("Step 4: Synthesizing final answer")
            synth_prompt = (
                f"Task: {task.description}\n\n"
                f"Analysis:\n{analysis}\n\n"
                "Based on the analysis above, provide a clear, concise, and accurate final answer."
            )
            answer, tokens = await self._llm.invoke(
                [
                    SystemMessage(content="You are a business analyst. Provide precise answers."),
                    HumanMessage(content=synth_prompt),
                ]
            )
            total_tokens += tokens
            retries += self._llm.last_retries

            # Fix 3: success=False when tools were called but every one failed
            success = tool_calls_total == 0 or tool_calls_successful > 0

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


def _parse_plan(text: str) -> tuple[list[dict], bool]:
    """Parse the LLM's JSON plan.

    Returns (queries, parse_ok). parse_ok=False means JSON decode failed.
    Tolerates leading/trailing markdown code fences.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
    try:
        data = json.loads(text)
        return data.get("queries", []), True
    except json.JSONDecodeError:
        return [], False
