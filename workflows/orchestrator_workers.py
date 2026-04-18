"""Workflow 6: Orchestrator-Workers — central planner delegates to specialized workers.

Pattern:
  Orchestrator — creates an execution plan (list of subtasks with worker assignments)
  Workers      — sql_worker, analysis_worker, search_worker, data_worker execute their subtask
  Orchestrator — aggregates all worker outputs and produces the final answer
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from agents.tool_dispatcher import TOOL_SCHEMAS, ToolDispatcher
from tasks.task_registry import Task
from workflows._utils import parse_json
from workflows.base import BaseWorkflow, WorkflowResult

log = logging.getLogger(__name__)

# Tools each worker type is allowed to use
_WORKER_TOOLS: dict[str, list[str]] = {
    "sql_worker": ["sql_query"],
    "analysis_worker": ["python_analysis", "calculator"],
    "search_worker": ["vector_search"],
    "data_worker": ["csv_reader"],
}

_ORCHESTRATOR_SYSTEM = (
    "You are an orchestrator agent. Given a business question, create a concrete execution plan "
    "by assigning subtasks to specialized workers.\n\n"
    "Available workers:\n"
    "- sql_worker: executes SQL queries against the business database\n"
    "- analysis_worker: runs Python/pandas analysis and math calculations\n"
    "- search_worker: searches business documents and policies\n"
    "- data_worker: reads raw CSV data files\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"subtasks": [{"worker": "<type>", "task": "<instruction>", "tool": "<tool_name>", "args": {...}}, ...]}\n'
    "Include 2–4 subtasks. Make args specific and executable — include full SQL queries or Python code."
)


class OrchestratorWorkersWorkflow(BaseWorkflow):
    """Central orchestrator delegates subtasks to specialized workers, then synthesizes."""

    name = "orchestrator_workers"
    description = "Orchestrator plans and delegates to specialized worker agents"

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
            # ── Orchestrator: Plan ─────────────────────────────────────────
            reasoning.append("Orchestrator: creating execution plan")
            plan_text, tokens = await self._llm.invoke(
                [
                    SystemMessage(content=_ORCHESTRATOR_SYSTEM),
                    HumanMessage(content=task.description),
                ]
            )
            total_tokens += tokens
            retries += self._llm.last_retries
            subtasks, parse_ok = _parse_subtasks(plan_text)

            # Fix 5: fail fast when plan is unparseable or empty — a hallucinated
            # fallback query produces garbage data and a confidently wrong answer.
            if not parse_ok:
                log.warning(
                    "Orchestrator plan parsing failed for task %s — aborting", task.id
                )
                reasoning.append(
                    "WARNING: Plan parsing failed (LLM returned non-JSON) — cannot execute"
                )
                latency_ms = (time.perf_counter() - start) * 1000
                return WorkflowResult(
                    task_id=task.id,
                    workflow_name=self.name,
                    success=False,
                    reasoning_steps=reasoning,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error="Orchestrator plan parsing failed",
                )
            if not subtasks:
                reasoning.append("WARNING: LLM returned empty subtask plan — cannot execute")
                latency_ms = (time.perf_counter() - start) * 1000
                return WorkflowResult(
                    task_id=task.id,
                    workflow_name=self.name,
                    success=False,
                    reasoning_steps=reasoning,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error="Orchestrator returned empty plan",
                )

            reasoning.append(f"Orchestrator: planned {len(subtasks)} subtasks across workers")

            # ── Workers: Execute concurrently ──────────────────────────────
            reasoning.append("Workers: executing subtasks concurrently")
            tool_calls_total += len(subtasks)
            for st in subtasks:
                tools_used.append(st.get("tool", st.get("worker", "unknown")))

            worker_results = await asyncio.gather(
                *[self._run_worker(st) for st in subtasks],
                return_exceptions=True,
            )

            worker_outputs: list[str] = []
            for st, res in zip(subtasks, worker_results):
                worker_name = st.get("worker", "worker")
                subtask_label = st.get("task", "")
                if isinstance(res, Exception):
                    worker_outputs.append(f"[{worker_name}] ERROR: {res}")
                    reasoning.append(f"  Worker {worker_name} raised exception: {res}")
                else:
                    tool_result, worker_tokens = res
                    total_tokens += worker_tokens
                    if tool_result.success:
                        tool_calls_successful += 1
                        output = self._dispatcher.result_to_text(tool_result)
                        worker_outputs.append(f"[{worker_name}: {subtask_label}]\n{output}")
                        reasoning.append(f"  Worker {worker_name} completed successfully")
                    else:
                        worker_outputs.append(
                            f"[{worker_name}: {subtask_label}] ERROR: {tool_result.error}"
                        )
                        reasoning.append(f"  Worker {worker_name} failed: {tool_result.error}")

            combined = "\n\n".join(worker_outputs)

            # ── Orchestrator: Synthesize ───────────────────────────────────
            reasoning.append("Orchestrator: synthesizing worker outputs into final answer")
            synth_prompt = (
                f"Question: {task.description}\n\n"
                f"Worker outputs:\n{combined}\n\n"
                "Synthesize all worker outputs into a comprehensive, accurate final answer."
            )
            answer, tokens = await self._llm.invoke(
                [
                    SystemMessage(content="You are a senior business analyst. Provide precise answers."),
                    HumanMessage(content=synth_prompt),
                ]
            )
            total_tokens += tokens
            retries += self._llm.last_retries

            # Fix 3: success=False when every tool call failed
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

    async def _run_worker(self, subtask: dict) -> tuple:
        """Execute a single worker subtask. Returns (ToolResult, tokens_used)."""
        tool_name = subtask.get("tool", "")
        args = subtask.get("args", {})

        if tool_name and args:
            result = await self._dispatcher.dispatch(tool_name, args)
            return result, 0

        worker_type = subtask.get("worker", "sql_worker")
        allowed = _WORKER_TOOLS.get(worker_type, ["sql_query"])
        tool_name, args, tokens = await self._worker_plan(subtask.get("task", ""), allowed)
        result = await self._dispatcher.dispatch(tool_name, args)
        return result, tokens

    async def _worker_plan(
        self, task_desc: str, allowed_tools: list[str]
    ) -> tuple[str, dict, int]:
        """Ask a worker LLM to pick a tool and construct args for its subtask."""
        allowed_schemas = [s for s in TOOL_SCHEMAS if s["name"] in allowed_tools]
        schemas_text = json.dumps(allowed_schemas, indent=2)
        prompt = (
            f"Subtask: {task_desc}\n\n"
            f"Available tools:\n{schemas_text}\n\n"
            'Respond with ONLY JSON: {"tool": "<tool_name>", "args": {...}}'
        )
        text, tokens = await self._llm.invoke(
            [
                SystemMessage(content="You are a specialized worker. Output only valid JSON."),
                HumanMessage(content=prompt),
            ]
        )
        # Fix 1: log worker plan parse failure
        parsed, parse_ok = parse_json(text)
        if not parse_ok:
            log.warning("Worker plan parsing failed for subtask '%s'", task_desc[:60])
        return (
            parsed.get("tool", allowed_tools[0]),
            parsed.get("args", {}),
            tokens,
        )


def _parse_subtasks(text: str) -> tuple[list[dict], bool]:
    """Parse orchestrator plan. Returns (subtasks, parse_ok)."""
    data, parse_ok = parse_json(text)
    return data.get("subtasks", []), parse_ok
