"""Workflow 8: Manager Agent — dynamic task coordination with active oversight.

Pattern:
  Manager — receives the task, plans which agents to use
  Agents  — specialized agents (sql, analysis, search, data) execute subtasks
  Manager — reviews each result; can reassign failed or low-quality subtasks
            to a different agent before synthesizing the final answer

Difference from Orchestrator-Workers (W6):
  - Orchestrator plans upfront and runs everything in parallel (static delegation)
  - Manager runs agents sequentially with active oversight; it sees each result
    and decides whether to accept it or delegate to a fallback agent
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from agents.tool_dispatcher import TOOL_SCHEMAS, ToolDispatcher
from tasks.task_registry import Task
from workflows._utils import MAX_TOOL_RESULT_CHARS, parse_json
from workflows.base import BaseWorkflow, WorkflowResult

log = logging.getLogger(__name__)

_MAX_AGENTS = 4   # maximum subtasks the manager can spawn
_MAX_RETRIES_PER_SUBTASK = 1  # how many times a failed subtask can be reassigned

# Tools each agent type is allowed to use
_AGENT_TOOLS: dict[str, list[str]] = {
    "sql_agent":      ["sql_query"],
    "analysis_agent": ["python_analysis", "calculator"],
    "search_agent":   ["vector_search"],
    "data_agent":     ["csv_reader"],
}

_MANAGER_PLAN_SYSTEM = (
    "You are a manager agent coordinating a team of specialists to answer a business question.\n\n"
    "Available agents:\n"
    "- sql_agent: executes SQL queries against the business database\n"
    "- analysis_agent: runs Python/pandas analysis and math calculations\n"
    "- search_agent: searches business documents and policies\n"
    "- data_agent: reads raw CSV data files\n\n"
    "Create a sequential execution plan — agents run one at a time so you can review each result.\n"
    'Respond with ONLY valid JSON:\n'
    '{"plan": [{"agent": "<type>", "task": "<specific instruction>", "tool": "<tool_name>", "args": {...}}, ...]}\n'
    "Include 2–4 subtasks. Make args specific and fully executable."
)

_MANAGER_REVIEW_SYSTEM = (
    "You are a manager reviewing an agent's output. "
    "Decide whether to accept the result or delegate to a fallback agent.\n\n"
    'Respond with ONLY valid JSON: {"accept": true/false, "reason": "<brief explanation>"}\n'
    "Accept if the output contains useful data. Reject only if it is empty or clearly wrong."
)

_MANAGER_FALLBACK_SYSTEM = (
    "You are a manager. The previous agent failed. "
    "Assign this subtask to a different agent type.\n\n"
    "Available agents: sql_agent, analysis_agent, search_agent, data_agent\n"
    'Respond with ONLY valid JSON:\n'
    '{"agent": "<type>", "tool": "<tool_name>", "args": {...}}'
)


class ManagerAgentWorkflow(BaseWorkflow):
    """Manager coordinates agents sequentially with active output review."""

    name = "manager_agent"
    description = "Manager reviews each agent output and re-delegates failed subtasks"

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
            # ── Manager: Plan ──────────────────────────────────────────────
            reasoning.append("Manager: creating sequential execution plan")
            plan_text, tokens = await self._llm.invoke([
                SystemMessage(content=_MANAGER_PLAN_SYSTEM),
                HumanMessage(content=task.description),
            ])
            total_tokens += tokens
            retries += self._llm.last_retries

            plan, parse_ok = _parse_plan(plan_text)
            if not parse_ok:
                log.warning("Manager plan parsing failed for task %s", task.id)
                reasoning.append("WARNING: Plan parsing failed — cannot execute")
                latency_ms = (time.perf_counter() - start) * 1000
                return WorkflowResult(
                    task_id=task.id,
                    workflow_name=self.name,
                    reasoning_steps=reasoning,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error="Manager plan parsing failed",
                )
            if not plan:
                reasoning.append("WARNING: Manager returned empty plan")
                latency_ms = (time.perf_counter() - start) * 1000
                return WorkflowResult(
                    task_id=task.id,
                    workflow_name=self.name,
                    reasoning_steps=reasoning,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error="Manager returned empty plan",
                )

            plan = plan[:_MAX_AGENTS]  # enforce cap
            reasoning.append(f"Manager: planned {len(plan)} subtasks")

            # ── Execute subtasks sequentially with oversight ───────────────
            agent_outputs: list[str] = []

            for idx, subtask in enumerate(plan):
                agent_name = subtask.get("agent", "sql_agent")
                subtask_desc = subtask.get("task", "")
                reasoning.append(f"Manager: delegating subtask {idx + 1} to {agent_name}")

                tool_result, tokens_used = await self._run_agent(subtask)
                total_tokens += tokens_used
                tool_calls_total += 1
                tools_used.append(subtask.get("tool", agent_name))

                # ── Manager reviews the result ─────────────────────────────
                result_text = self._dispatcher.result_to_text(tool_result)[:MAX_TOOL_RESULT_CHARS]
                review_ok, review_reason, rev_tokens = await self._review_result(
                    subtask_desc, result_text, tool_result.success
                )
                total_tokens += rev_tokens
                retries += self._llm.last_retries

                if review_ok and tool_result.success:
                    tool_calls_successful += 1
                    reasoning.append(f"  Manager accepted result: {review_reason[:80]}")
                    agent_outputs.append(f"[{agent_name}: {subtask_desc}]\n{result_text}")
                else:
                    reasoning.append(
                        f"  Manager rejected result: {review_reason[:80]} — trying fallback"
                    )

                    # Attempt fallback reassignment once
                    fallback, fb_tokens = await self._get_fallback(subtask_desc, agent_name)
                    total_tokens += fb_tokens
                    if fallback:
                        fallback_result, fb_tool_tokens = await self._run_agent(fallback)
                        total_tokens += fb_tool_tokens
                        tool_calls_total += 1
                        tools_used.append(fallback.get("tool", fallback.get("agent", "unknown")))

                        fb_text = self._dispatcher.result_to_text(fallback_result)[:MAX_TOOL_RESULT_CHARS]
                        if fallback_result.success:
                            tool_calls_successful += 1
                            reasoning.append("  Fallback agent succeeded")
                            agent_outputs.append(
                                f"[{fallback.get('agent', 'fallback')}: {subtask_desc}]\n{fb_text}"
                            )
                        else:
                            reasoning.append("  Fallback agent also failed")
                            agent_outputs.append(
                                f"[{agent_name}: {subtask_desc}] ERROR: both primary and fallback failed"
                            )
                    else:
                        agent_outputs.append(
                            f"[{agent_name}: {subtask_desc}] ERROR: {tool_result.error or 'rejected'}"
                        )

            combined = "\n\n".join(agent_outputs)

            # ── Manager: Synthesize final answer ───────────────────────────
            reasoning.append("Manager: synthesizing final answer from all agent outputs")
            synth_prompt = (
                f"Question: {task.description}\n\n"
                f"Agent outputs:\n{combined}\n\n"
                "Synthesize all agent outputs into a comprehensive, accurate final answer."
            )
            answer, tokens = await self._llm.invoke([
                SystemMessage(content="You are a senior business analyst. Provide precise answers."),
                HumanMessage(content=synth_prompt),
            ])
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

    async def _run_agent(self, subtask: dict) -> tuple:
        """Execute a subtask. Returns (ToolResult, tokens_used)."""
        tool_name = subtask.get("tool", "")
        args = subtask.get("args", {})

        if tool_name and args:
            result = await self._dispatcher.dispatch(tool_name, args)
            return result, 0

        # Agent needs to self-plan
        agent_type = subtask.get("agent", "sql_agent")
        allowed = _AGENT_TOOLS.get(agent_type, ["sql_query"])
        tool_name, args, tokens = await self._agent_self_plan(subtask.get("task", ""), allowed)
        result = await self._dispatcher.dispatch(tool_name, args)
        return result, tokens

    async def _agent_self_plan(
        self, task_desc: str, allowed_tools: list[str]
    ) -> tuple[str, dict, int]:
        """Agent selects a tool and constructs args for its subtask."""
        allowed_schemas = [s for s in TOOL_SCHEMAS if s["name"] in allowed_tools]
        prompt = (
            f"Subtask: {task_desc}\n\n"
            f"Available tools: {json.dumps(allowed_schemas, indent=2)}\n\n"
            'Respond with ONLY JSON: {"tool": "<tool_name>", "args": {...}}'
        )
        text, tokens = await self._llm.invoke([
            SystemMessage(content="You are a specialized agent. Output only valid JSON."),
            HumanMessage(content=prompt),
        ])
        data, parse_ok = parse_json(text)
        if not parse_ok:
            log.warning("Agent self-plan failed for subtask '%.60s'", task_desc)
        return data.get("tool", allowed_tools[0]), data.get("args", {}), tokens

    async def _review_result(
        self, subtask_desc: str, result_text: str, tool_success: bool
    ) -> tuple[bool, str, int]:
        """Manager reviews an agent result. Returns (accept, reason, tokens)."""
        if not tool_success:
            return False, "Tool call failed", 0

        review_prompt = (
            f"Subtask: {subtask_desc}\n\n"
            f"Agent output:\n{result_text[:1000]}\n\n"
            "Is this output useful for answering the subtask?"
        )
        text, tokens = await self._llm.invoke([
            SystemMessage(content=_MANAGER_REVIEW_SYSTEM),
            HumanMessage(content=review_prompt),
        ])
        data, parse_ok = parse_json(text)
        if not parse_ok:
            # Default to accepting on parse failure to avoid infinite loops
            return True, "Parse failed — accepting by default", tokens
        accept = bool(data.get("accept", True))
        reason = str(data.get("reason", ""))
        return accept, reason, tokens

    async def _get_fallback(
        self, subtask_desc: str, failed_agent: str
    ) -> tuple[dict | None, int]:
        """Ask manager for a fallback agent assignment."""
        prompt = (
            f"Subtask: {subtask_desc}\n\n"
            f"The {failed_agent} failed. Assign to a different agent."
        )
        text, tokens = await self._llm.invoke([
            SystemMessage(content=_MANAGER_FALLBACK_SYSTEM),
            HumanMessage(content=prompt),
        ])
        data, parse_ok = parse_json(text)
        if not parse_ok or not data.get("agent"):
            return None, tokens
        return {"agent": data["agent"], "tool": data.get("tool", ""), "args": data.get("args", {}), "task": subtask_desc}, tokens


def _parse_plan(text: str) -> tuple[list[dict], bool]:
    data, ok = parse_json(text)
    return data.get("plan", []), ok
