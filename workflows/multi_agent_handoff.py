"""Workflow 9: Multi-Agent Handoff — sequential agent pipeline with context passing.

Pattern:
  Researcher  — retrieves raw data from databases and documents
  Analyst     — interprets and analyzes the retrieved data
  Writer      — formats the final answer for the business audience

Each agent receives the full output of all previous agents.  Unlike the
Orchestrator (parallel) or Manager (sequential-with-oversight), handoffs are
one-directional: each agent enriches the shared context before passing it on.

This mirrors the "assembly-line" handoff pattern where each specialist adds
value and passes forward — no agent can send work backwards.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from agents.tool_dispatcher import TOOL_SCHEMAS, ToolDispatcher
from tasks.task_registry import Task
from workflows._utils import MAX_TOOL_RESULT_CHARS
from workflows.base import BaseWorkflow, WorkflowResult

log = logging.getLogger(__name__)

_MAX_RESEARCHER_TOOLS = 3

_RESEARCHER_SYSTEM = (
    "You are a data researcher. Your job is ONLY to retrieve raw data — no analysis.\n"
    "Use the available tools to gather all data relevant to the question.\n"
    "After using tools, summarize ONLY the raw facts and numbers you found."
)

_ANALYST_SYSTEM = (
    "You are a business analyst. The researcher has gathered raw data for you.\n"
    "Analyze this data: identify trends, calculate key metrics, spot anomalies.\n"
    "Do not repeat the raw data — provide analytical insights and calculations."
)

_WRITER_SYSTEM = (
    "You are a business communication specialist. "
    "The analyst has produced data insights. Turn these into a clear, concise, "
    "well-structured answer for a business stakeholder.\n"
    "Focus on actionable conclusions. Use bullet points where helpful."
)


class MultiAgentHandoffWorkflow(BaseWorkflow):
    """Three-agent handoff: Researcher → Analyst → Writer."""

    name = "multi_agent_handoff"
    description = "Sequential Researcher → Analyst → Writer handoff, each enriching shared context"

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
            # ── Agent 1: Researcher ────────────────────────────────────────
            reasoning.append("Handoff 1/3: Researcher retrieving raw data")
            researcher_output, r_tools_used, r_tool_total, r_tool_ok, r_tokens, r_retries = (
                await self._run_researcher(task.description)
            )
            tools_used.extend(r_tools_used)
            tool_calls_total += r_tool_total
            tool_calls_successful += r_tool_ok
            total_tokens += r_tokens
            retries += r_retries
            reasoning.append(
                f"  Researcher used {r_tool_total} tool(s) ({r_tool_ok} succeeded), "
                f"produced {len(researcher_output)} chars"
            )

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
                    error="Researcher: all tool calls failed",
                )

            # ── Agent 2: Analyst ───────────────────────────────────────────
            reasoning.append("Handoff 2/3: Analyst interpreting researcher output")
            analyst_prompt = (
                f"Business question: {task.description}\n\n"
                f"Researcher's raw data:\n{researcher_output}\n\n"
                "Analyze this data and provide business insights."
            )
            analyst_output, tokens = await self._llm.invoke([
                SystemMessage(content=_ANALYST_SYSTEM),
                HumanMessage(content=analyst_prompt),
            ])
            total_tokens += tokens
            retries += self._llm.last_retries
            reasoning.append(f"  Analyst produced {len(analyst_output)} chars of analysis")

            # ── Agent 3: Writer ────────────────────────────────────────────
            reasoning.append("Handoff 3/3: Writer formatting final answer")
            writer_prompt = (
                f"Business question: {task.description}\n\n"
                f"Researcher data:\n{researcher_output}\n\n"
                f"Analyst insights:\n{analyst_output}\n\n"
                "Produce the final business answer."
            )
            final_answer, tokens = await self._llm.invoke([
                SystemMessage(content=_WRITER_SYSTEM),
                HumanMessage(content=writer_prompt),
            ])
            total_tokens += tokens
            retries += self._llm.last_retries
            reasoning.append("Writer produced final answer")

            success = tool_calls_total == 0 or tool_calls_successful > 0

            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                answer=final_answer,
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

    async def _run_researcher(
        self, task_description: str
    ) -> tuple[str, list[str], int, int, int, int]:
        """Run the researcher agent.  Returns (output, tools_used, total_calls, successful_calls, tokens, retries)."""
        from langchain_core.messages import AIMessage, ToolMessage
        from agents.llm_client import extract_text

        messages = [
            SystemMessage(content=_RESEARCHER_SYSTEM),
            HumanMessage(content=(
                f"Task: {task_description}\n\n"
                "Retrieve all data needed to answer this question. Use tools to get real data."
            )),
        ]

        tools_used: list[str] = []
        total_calls = 0
        successful_calls = 0
        total_tokens = 0
        retries = 0
        budget = _MAX_RESEARCHER_TOOLS

        for _ in range(budget + 2):
            response, tokens = await self._llm.invoke_with_tools(messages, TOOL_SCHEMAS)
            total_tokens += tokens
            retries += self._llm.last_retries

            if not response.tool_calls or budget <= 0:
                output = extract_text(response)
                return output, tools_used, total_calls, successful_calls, total_tokens, retries

            messages.append(response)
            for tc in response.tool_calls:
                tool_name = tc.get("name", "") if isinstance(tc, dict) else tc.name
                tool_args = tc.get("args", {}) if isinstance(tc, dict) else tc.args
                tool_id = tc.get("id", tool_name) if isinstance(tc, dict) else tc.id

                tools_used.append(tool_name)
                total_calls += 1
                budget -= 1

                result = await self._dispatcher.dispatch(tool_name, tool_args)
                result_text = self._dispatcher.result_to_text(result)[:MAX_TOOL_RESULT_CHARS]

                if result.success:
                    successful_calls += 1

                messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))

        # Fallback: ask the model to summarize what it knows
        output, tokens = await self._llm.invoke([
            SystemMessage(content=_RESEARCHER_SYSTEM),
            HumanMessage(content=f"Summarize the data you retrieved for: {task_description}"),
        ])
        total_tokens += tokens
        return output, tools_used, total_calls, successful_calls, total_tokens, retries
