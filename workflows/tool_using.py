"""Workflow 2: Tool-using agent — ReAct loop with all available tools."""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.llm_client import LLMClient, extract_text  # Fix 6: import shared extract_text
from agents.tool_dispatcher import ToolDispatcher
from tasks.task_registry import Task
from workflows.base import BaseWorkflow, WorkflowResult

_SYSTEM = (
    "You are a business intelligence agent with access to tools. "
    "Use the available tools to answer the question accurately. "
    "Think step by step: decide which tools to use, call them, inspect results, "
    "and synthesize a final answer. "
    "When you have enough information, respond with your final answer — do NOT call more tools."
)

_MAX_ITERATIONS = 10
_MAX_TOOL_RESULT_CHARS = 3000  # Fix 8: cap individual tool results to bound context growth


class ToolUsingWorkflow(BaseWorkflow):
    """ReAct-style agent that can call any available tool iteratively."""

    name = "tool_using"
    description = "LLM with iterative tool access (ReAct loop)"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._dispatcher = ToolDispatcher()

    async def run(self, task: Task) -> WorkflowResult:
        # Fix 11: validate task before doing any LLM work
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
            messages: list = [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=task.description),
            ]
            reasoning.append("Initiated tool-using agent with task")

            for iteration in range(_MAX_ITERATIONS):
                response, tokens = await self._llm.invoke_with_tools(
                    messages, self._dispatcher.schemas
                )
                total_tokens += tokens
                retries += self._llm.last_retries
                messages.append(response)

                if not response.tool_calls:
                    reasoning.append(f"Final answer generated after {iteration + 1} iteration(s)")
                    answer = extract_text(response)
                    latency_ms = (time.perf_counter() - start) * 1000
                    return WorkflowResult(
                        task_id=task.id,
                        workflow_name=self.name,
                        answer=answer,
                        success=True,
                        reasoning_steps=reasoning,
                        tools_used=list(dict.fromkeys(tools_used)),
                        tool_calls_total=tool_calls_total,
                        tool_calls_successful=tool_calls_successful,
                        total_tokens=total_tokens,
                        latency_ms=latency_ms,
                        retries=retries,
                    )

                # Execute all tool calls in this iteration
                for tc in response.tool_calls:
                    # Fix 2: use .get() with safe defaults instead of direct key access
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})
                    tool_id = tc.get("id", "")

                    if not tool_name:
                        reasoning.append("Skipping malformed tool call (missing name)")
                        continue

                    reasoning.append(f"Calling tool: {tool_name}({_fmt_args(tool_args)})")
                    tools_used.append(tool_name)
                    tool_calls_total += 1

                    result = await self._dispatcher.dispatch(tool_name, tool_args)
                    if result.success:
                        tool_calls_successful += 1
                    else:
                        reasoning.append(f"Tool {tool_name} failed: {result.error}")

                    # Fix 8: truncate large tool results before adding to message history
                    tool_text = self._dispatcher.result_to_text(result)
                    if len(tool_text) > _MAX_TOOL_RESULT_CHARS:
                        tool_text = (
                            tool_text[:_MAX_TOOL_RESULT_CHARS]
                            + f"\n... [truncated {len(tool_text) - _MAX_TOOL_RESULT_CHARS} chars]"
                        )
                    messages.append(ToolMessage(content=tool_text, tool_call_id=tool_id))

            # Hit iteration limit — degraded path, mark as failure
            reasoning.append(
                f"Max iterations ({_MAX_ITERATIONS}) reached — agent could not reach a "
                "conclusion within the allowed budget"
            )
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
                error=f"Iteration limit ({_MAX_ITERATIONS}) exceeded without reaching a final answer",
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


def _fmt_args(args: dict) -> str:
    """Short representation of tool args for reasoning logs."""
    short = {k: (str(v)[:60] + "..." if len(str(v)) > 60 else v) for k, v in args.items()}
    return str(short)
