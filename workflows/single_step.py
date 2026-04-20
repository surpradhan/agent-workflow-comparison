"""Workflow 1: Single-step LLM — no tools, single LLM call."""

from __future__ import annotations

import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from tasks.task_registry import Task
from workflows.base import BaseWorkflow, WorkflowResult

_SYSTEM = (
    "You are a senior business analyst. "
    "Answer the following business question accurately and concisely, "
    "using your knowledge of typical business operations, KPIs, and analytics."
)


class SingleStepWorkflow(BaseWorkflow):
    """Direct LLM call — no tools, no iteration, pure language model reasoning."""

    name = "single_step"
    description = "Direct LLM call without tool access"

    def __init__(self) -> None:
        self._llm = LLMClient()

    async def run(self, task: Task) -> WorkflowResult:
        if err := self._validate_task(task):
            return WorkflowResult(
                task_id=task.id, workflow_name=self.name, success=False, error=err
            )

        start = time.perf_counter()
        try:
            messages = [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=task.description),
            ]
            answer, tokens = await self._llm.invoke(messages)
            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                answer=answer,
                success=True,
                reasoning_steps=["Single LLM call with task description", "Answer generated"],
                tools_used=[],
                tool_calls_total=0,
                tool_calls_successful=0,
                total_tokens=tokens,
                latency_ms=latency_ms,
                retries=self._llm.last_retries,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                success=False,
                latency_ms=latency_ms,
                error=str(exc),
            )
