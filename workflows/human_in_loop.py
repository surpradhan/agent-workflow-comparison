"""Workflow 10: Human-in-the-Loop — supervised execution with review checkpoints.

Pattern:
  Plan      — LLM proposes a retrieval plan and explains its reasoning
  Review    — (simulated) human approves, rejects, or edits the plan
  Execute   — approved plan is executed against real data tools
  Confirm   — (simulated) human reviews the draft answer before finalization

In benchmarking mode (``simulate_human=True``, the default), the human review
is performed by a separate LLM call acting as a "critical but fair" reviewer.
This preserves the structural checkpoints — which affect latency, token cost,
and reasoning-step counts — while allowing fully automated benchmarking.

In a production deployment, replace ``_simulated_human_review`` and
``_simulated_human_confirm`` with calls to your preferred human-in-the-loop
platform (e.g. a Slack bot, a web UI approval queue, or a LangSmith annotation
workflow).

Simulated reviewer behaviour:
- Plan review: approves if the plan is coherent; asks for clarification otherwise
- Answer confirm: approves if the answer is responsive; flags gaps otherwise
"""

from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from agents.tool_dispatcher import ToolDispatcher
from tasks.task_registry import Task
from workflows._utils import MAX_TOOL_RESULT_CHARS, parse_json
from workflows.base import BaseWorkflow, WorkflowResult

log = logging.getLogger(__name__)

_MAX_PLAN_QUERIES = 3
_MAX_PLAN_REVISIONS = 1   # how many times the human can send the plan back

_PLANNER_SYSTEM = (
    "You are a business analyst planning how to answer a business question.\n\n"
    "Propose a clear step-by-step retrieval plan before executing anything.\n"
    "Be explicit about which tools you will use and why.\n\n"
    'Respond with ONLY valid JSON:\n'
    '{"rationale": "<why these steps>", '
    '"queries": [{"tool": "<name>", "args": {...}, "purpose": "<why>"}]}\n'
    "Use tools: sql_query, calculator, vector_search, csv_reader, python_analysis. "
    f"Include at most {_MAX_PLAN_QUERIES} queries."
)

_HUMAN_REVIEW_SYSTEM = (
    "You are a domain expert reviewing an analyst's retrieval plan.\n\n"
    "Approve the plan if the proposed queries will gather the data needed to answer the question.\n"
    "Reject it (with specific feedback) only if a critical data source is missing.\n\n"
    'Respond with ONLY valid JSON: {"approved": true/false, "feedback": "<comment or empty>"}'
)

_HUMAN_CONFIRM_SYSTEM = (
    "You are a business stakeholder reviewing a draft answer.\n\n"
    "Approve the answer if it is accurate and responsive to the question.\n"
    "Reject it (with specific feedback) only if it is clearly wrong or missing critical information.\n\n"
    'Respond with ONLY valid JSON: {"approved": true/false, "feedback": "<comment or empty>"}'
)

_ANALYST_SYSTEM = "You are a business analyst. Provide precise, data-backed answers."


class HumanInLoopWorkflow(BaseWorkflow):
    """Human-reviewed planning and answer confirmation (simulated in benchmarks)."""

    name = "human_in_loop"
    description = "Human checkpoints at plan review and answer confirmation (simulated for benchmarks)"

    def __init__(self, simulate_human: bool = True) -> None:
        """
        Args:
            simulate_human: When True (default), use an LLM to simulate human review.
                            Set to False when integrating with a real human-in-the-loop
                            platform — you must then implement the ``_human_*`` hooks.
        """
        self._llm = LLMClient()
        self._dispatcher = ToolDispatcher()
        self.simulate_human = simulate_human

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
            # ── Checkpoint 1: Propose and review plan ──────────────────────
            reasoning.append("Checkpoint 1: Analyst proposing retrieval plan")
            plan_text, tokens = await self._llm.invoke([
                SystemMessage(content=_PLANNER_SYSTEM),
                HumanMessage(content=task.description),
            ])
            total_tokens += tokens
            retries += self._llm.last_retries

            queries, rationale, parse_ok = _parse_plan(plan_text)
            if not parse_ok:
                log.warning("Plan parsing failed for task %s", task.id)
                reasoning.append("WARNING: Plan parsing failed — cannot proceed")
                latency_ms = (time.perf_counter() - start) * 1000
                return WorkflowResult(
                    task_id=task.id,
                    workflow_name=self.name,
                    success=False,
                    reasoning_steps=reasoning,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error="Plan parsing failed",
                )

            reasoning.append(f"  Analyst rationale: {rationale[:120]}")
            reasoning.append(f"  Plan: {len(queries)} queries proposed")

            # ── Human plan review (with optional revision) ─────────────────
            for revision in range(_MAX_PLAN_REVISIONS + 1):
                approved, feedback, rev_tokens = await self._human_plan_review(
                    task.description, queries, rationale
                )
                total_tokens += rev_tokens
                reasoning.append(
                    f"  Human plan review {'approved' if approved else 'rejected'}"
                    + (f": {feedback[:100]}" if feedback else "")
                )

                if approved:
                    break

                if revision == _MAX_PLAN_REVISIONS:
                    reasoning.append("  Max plan revisions reached — proceeding with current plan")
                    break

                # Analyst revises the plan based on feedback
                reasoning.append(f"  Analyst revising plan based on feedback: {feedback[:100]}")
                revise_prompt = (
                    f"Original task: {task.description}\n\n"
                    f"Your proposed plan was rejected with this feedback:\n{feedback}\n\n"
                    "Revise your retrieval plan to address the feedback.\n"
                    'Respond with ONLY valid JSON: {"rationale": "...", "queries": [...]}'
                )
                plan_text, tokens = await self._llm.invoke([
                    SystemMessage(content=_PLANNER_SYSTEM),
                    HumanMessage(content=revise_prompt),
                ])
                total_tokens += tokens
                retries += self._llm.last_retries
                queries, rationale, parse_ok = _parse_plan(plan_text)
                if not parse_ok:
                    reasoning.append("  Revision plan parsing failed — using original plan")
                    break

            # ── Execute approved plan ──────────────────────────────────────
            reasoning.append("Executing approved plan")
            retrieved_parts: list[str] = []
            for item in queries[:_MAX_PLAN_QUERIES]:
                tool_name = item.get("tool", "")
                args = item.get("args", {})
                purpose = item.get("purpose", "")
                tools_used.append(tool_name)
                tool_calls_total += 1

                result = await self._dispatcher.dispatch(tool_name, args)
                result_text = self._dispatcher.result_to_text(result)[:MAX_TOOL_RESULT_CHARS]

                if result.success:
                    tool_calls_successful += 1
                    retrieved_parts.append(f"[{tool_name} — {purpose}]\n{result_text}")
                    reasoning.append(f"  {tool_name}: ok")
                else:
                    retrieved_parts.append(f"[{tool_name}] ERROR: {result.error}")
                    reasoning.append(f"  {tool_name}: failed — {result.error}")

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

            retrieved_data = "\n\n".join(retrieved_parts) or "(no data retrieved)"

            # ── Draft answer ───────────────────────────────────────────────
            reasoning.append("Analyst drafting answer from retrieved data")
            draft_prompt = (
                f"Task: {task.description}\n\n"
                f"Retrieved data:\n{retrieved_data}\n\n"
                "Provide a comprehensive, accurate answer."
            )
            draft_answer, tokens = await self._llm.invoke([
                SystemMessage(content=_ANALYST_SYSTEM),
                HumanMessage(content=draft_prompt),
            ])
            total_tokens += tokens
            retries += self._llm.last_retries
            reasoning.append("Draft answer produced")

            # ── Checkpoint 2: Human answer confirmation ────────────────────
            reasoning.append("Checkpoint 2: Human reviewing draft answer")
            answer_approved, answer_feedback, conf_tokens = await self._human_answer_confirm(
                task.description, draft_answer
            )
            total_tokens += conf_tokens
            reasoning.append(
                f"  Human answer review {'approved' if answer_approved else 'rejected'}"
                + (f": {answer_feedback[:100]}" if answer_feedback else "")
            )

            if not answer_approved and answer_feedback:
                # Analyst refines based on feedback — one revision pass
                reasoning.append("  Analyst refining answer based on human feedback")
                refine_prompt = (
                    f"Task: {task.description}\n\n"
                    f"Retrieved data:\n{retrieved_data}\n\n"
                    f"Your draft answer was flagged with this feedback:\n{answer_feedback}\n\n"
                    "Produce an improved final answer."
                )
                draft_answer, tokens = await self._llm.invoke([
                    SystemMessage(content=_ANALYST_SYSTEM),
                    HumanMessage(content=refine_prompt),
                ])
                total_tokens += tokens
                retries += self._llm.last_retries
                reasoning.append("  Refined answer produced")

            success = tool_calls_total == 0 or tool_calls_successful > 0

            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                answer=draft_answer,
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

    # ── Human review hooks ─────────────────────────────────────────────────

    async def _human_plan_review(
        self, task_desc: str, queries: list[dict], rationale: str
    ) -> tuple[bool, str, int]:
        """Return (approved, feedback, tokens)."""
        if not self.simulate_human:
            raise NotImplementedError(
                "Real human plan review is not wired up. "
                "Implement this method to integrate with your HITL platform."
            )
        return await self._simulated_human_review(
            task_desc, queries, rationale, _HUMAN_REVIEW_SYSTEM
        )

    async def _human_answer_confirm(
        self, task_desc: str, answer: str
    ) -> tuple[bool, str, int]:
        """Return (approved, feedback, tokens)."""
        if not self.simulate_human:
            raise NotImplementedError(
                "Real human answer confirmation is not wired up. "
                "Implement this method to integrate with your HITL platform."
            )
        review_prompt = (
            f"Question: {task_desc}\n\n"
            f"Draft answer:\n{answer}"
        )
        text, tokens = await self._llm.invoke([
            SystemMessage(content=_HUMAN_CONFIRM_SYSTEM),
            HumanMessage(content=review_prompt),
        ])
        approved, feedback = _parse_review_response(text, context="answer confirmation")
        return approved, feedback, tokens

    async def _simulated_human_review(
        self,
        task_desc: str,
        queries: list[dict],
        rationale: str,
        system_prompt: str,
    ) -> tuple[bool, str, int]:
        """LLM-simulated human reviewer."""
        plan_summary = json.dumps(queries, indent=2)
        review_prompt = (
            f"Question: {task_desc}\n\n"
            f"Analyst rationale: {rationale}\n\n"
            f"Proposed queries:\n{plan_summary}"
        )
        text, tokens = await self._llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=review_prompt),
        ])
        approved, feedback = _parse_review_response(text, context="plan review")
        return approved, feedback, tokens


def _parse_review_response(text: str, context: str = "review") -> tuple[bool, str]:
    """Parse a human-reviewer JSON response.

    Returns (approved, feedback).  When parsing fails the response is logged
    as a warning and approval is granted so the benchmark is not blocked —
    but the failure is visible in the log and can be caught in tests.
    """
    data, parse_ok = parse_json(text)
    if not parse_ok:
        log.warning(
            "Human reviewer (%s) returned unparseable JSON — defaulting to approval. "
            "Raw response: %.120s",
            context,
            text,
        )
        return True, ""
    return bool(data.get("approved", True)), str(data.get("feedback", ""))


def _parse_plan(text: str) -> tuple[list[dict], str, bool]:
    """Parse planner JSON. Returns (queries, rationale, parse_ok)."""
    data, ok = parse_json(text)
    return data.get("queries", []), data.get("rationale", ""), ok
