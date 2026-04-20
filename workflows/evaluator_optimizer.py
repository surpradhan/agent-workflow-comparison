"""Workflow 7: Evaluator-Optimizer — iterative self-refinement loop.

Pattern:
  Generator  — produces an initial answer (with tool access)
  Evaluator  — critiques the answer against the task requirements
  Optimizer  — refines the answer using the critique
  Loop       — repeats until quality threshold is met or max iterations reached

The evaluator scores each candidate answer 0–10 and identifies specific gaps.
The optimizer receives the original task, the current answer, and the critique
before producing an improved version.  This mirrors the "self-refine" pattern
from Madaan et al. (2023).
"""

from __future__ import annotations

import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.llm_client import LLMClient, extract_text
from agents.tool_dispatcher import TOOL_SCHEMAS, ToolDispatcher
from tasks.task_registry import Task
from workflows._utils import MAX_TOOL_RESULT_CHARS, parse_json
from workflows.base import BaseWorkflow, WorkflowResult

log = logging.getLogger(__name__)

_MAX_ITERATIONS = 3
_QUALITY_THRESHOLD = 8  # score out of 10; stop early if this is reached or exceeded
_MAX_TOOL_CALLS = 4     # total tool budget for the initial generation phase

_GENERATOR_SYSTEM = (
    "You are a business analyst with access to data tools. "
    "Answer the question thoroughly using available tools to retrieve real data. "
    "Be precise with numbers and cite your sources."
)

_EVALUATOR_SYSTEM = (
    "You are a critical evaluator assessing a business analyst's answer.\n\n"
    "Score the answer from 0 to 10 and identify specific deficiencies:\n"
    "  10 — Complete, accurate, well-supported by data, actionable\n"
    "   7 — Good but missing minor details or precision\n"
    "   4 — Partially answers but has significant gaps or errors\n"
    "   0 — Wrong, irrelevant, or unsupported\n\n"
    'Example: {"score": 7, "critique": "Missing Q4 breakdown and growth rate calculation."}\n\n'
    "Respond with ONLY a JSON object in exactly that format. "
    "No prose, no markdown fences, no explanation before or after."
)

_OPTIMIZER_SYSTEM = (
    "You are a business analyst refining your answer based on a critique. "
    "Address every point raised in the critique. "
    "Use the original data already retrieved — do not repeat tool calls unless data is missing. "
    "Produce a complete, improved answer."
)


class EvaluatorOptimizerWorkflow(BaseWorkflow):
    """Iterative generate → evaluate → optimize loop with early stopping."""

    name = "evaluator_optimizer"
    description = "Iterative self-refinement: generate, evaluate, and optimize until quality threshold"

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
            # ── Phase 1: Initial answer generation (with tools) ────────────
            reasoning.append("Phase 1: Generating initial answer with tool access")

            current_answer = ""  # initialised here; always set in the loop below
            messages = [
                SystemMessage(content=_GENERATOR_SYSTEM),
                HumanMessage(content=task.description),
            ]

            # ReAct-style tool loop (budget-limited).
            # Uses an explicit counter rather than for/else to avoid the
            # unreachable-else / undefined-variable pitfall.
            retrieved_context = ""
            budget_remaining = _MAX_TOOL_CALLS

            while True:
                response, tokens = await self._llm.invoke_with_tools(messages, TOOL_SCHEMAS)
                total_tokens += tokens
                retries += self._llm.last_retries

                if not response.tool_calls:
                    # Model chose to answer directly — we're done retrieving.
                    current_answer = extract_text(response)
                    break

                if budget_remaining <= 0:
                    # Budget exhausted — force a text answer without more tool calls.
                    finish_prompt = (
                        f"Task: {task.description}\n\n"
                        f"Retrieved data:\n{retrieved_context}\n\n"
                        "Provide a comprehensive answer based on the data above."
                    )
                    current_answer, tokens = await self._llm.invoke([
                        SystemMessage(content=_GENERATOR_SYSTEM),
                        HumanMessage(content=finish_prompt),
                    ])
                    total_tokens += tokens
                    retries += self._llm.last_retries
                    break

                # Execute each tool call the model requested.
                messages.append(response)
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "") if isinstance(tc, dict) else tc.name
                    tool_args = tc.get("args", {}) if isinstance(tc, dict) else tc.args
                    tool_id = tc.get("id", tool_name) if isinstance(tc, dict) else tc.id

                    tools_used.append(tool_name)
                    tool_calls_total += 1
                    budget_remaining -= 1

                    result = await self._dispatcher.dispatch(tool_name, tool_args)
                    result_text = self._dispatcher.result_to_text(result)[:MAX_TOOL_RESULT_CHARS]

                    if result.success:
                        tool_calls_successful += 1
                        retrieved_context += f"\n[{tool_name}]: {result_text}"
                        reasoning.append(f"  Tool {tool_name}: ok")
                    else:
                        reasoning.append(f"  Tool {tool_name}: failed — {result.error}")

                    messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))

            reasoning.append(f"Initial answer produced ({len(current_answer)} chars)")

            # ── Iterative evaluate → optimize loop ─────────────────────────
            # best_answer always holds the answer with the highest score seen.
            best_answer = current_answer
            best_score = 0

            for iteration in range(1, _MAX_ITERATIONS + 1):
                reasoning.append(f"Iteration {iteration}: evaluating answer quality")

                eval_prompt = (
                    f"Task: {task.description}\n\n"
                    f"Answer to evaluate:\n{current_answer}"
                )
                eval_text, tokens = await self._llm.invoke([
                    SystemMessage(content=_EVALUATOR_SYSTEM),
                    HumanMessage(content=eval_prompt),
                ])
                total_tokens += tokens
                retries += self._llm.last_retries

                score, critique = _parse_evaluation(eval_text)
                reasoning.append(f"  Score: {score}/10 — {critique[:120]}")

                # Update the high-water mark.
                if score > best_score:
                    best_score = score
                    best_answer = current_answer

                if score >= _QUALITY_THRESHOLD:
                    reasoning.append(
                        f"  Quality threshold reached ({score} >= {_QUALITY_THRESHOLD}) — stopping"
                    )
                    break

                if iteration == _MAX_ITERATIONS:
                    reasoning.append("  Max iterations reached — returning best answer")
                    break

                # Optimize: generate a refined answer for the next iteration.
                reasoning.append(f"Iteration {iteration}: optimizing based on critique")
                opt_prompt = (
                    f"Original task: {task.description}\n\n"
                    f"Your previous answer:\n{current_answer}\n\n"
                    f"Critique (score {score}/10):\n{critique}\n\n"
                    "Address every point in the critique and produce an improved, complete answer."
                )
                current_answer, tokens = await self._llm.invoke([
                    SystemMessage(content=_OPTIMIZER_SYSTEM),
                    HumanMessage(content=opt_prompt),
                ])
                total_tokens += tokens
                retries += self._llm.last_retries

            # best_answer is kept up-to-date inside the loop; no post-loop fixup needed.

            latency_ms = (time.perf_counter() - start) * 1000
            return WorkflowResult(
                task_id=task.id,
                workflow_name=self.name,
                answer=best_answer,
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


def _parse_evaluation(text: str) -> tuple[int, str]:
    """Parse evaluator JSON response. Returns (score, critique).

    Falls back to a regex heuristic when the LLM returns prose instead of JSON.
    """
    data, parse_ok = parse_json(text)
    if parse_ok:
        try:
            score = max(0, min(10, int(data.get("score", 0))))
            critique = str(data.get("critique", "No critique provided"))
            return score, critique
        except (ValueError, TypeError):
            pass

    log.warning("Evaluator response was not valid JSON: %.80s", text)
    m = re.search(r"\b(10|[0-9])\b", text)
    score = int(m.group(1)) if m else 0
    return score, text[:500]
