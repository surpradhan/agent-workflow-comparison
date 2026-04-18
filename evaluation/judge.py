"""LLM-as-judge evaluator for scoring workflow answers against ground truth.

Each workflow answer is scored 0.0–1.0 by a separate LLM call that is given
the task description, evaluation criteria, ground truth hint, and the agent's
answer.  Scoring is optional — if a task has no ground_truth and no
evaluation_criteria, score() returns None and the task is excluded from the
avg_quality_score metric.
"""

from __future__ import annotations

import asyncio
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient
from config import settings
from tasks.task_registry import Task

log = logging.getLogger(__name__)

# Maximum number of concurrent judge LLM calls.
# Keeps cloud-provider request rates within tier limits and prevents
# overwhelming a local Ollama instance (which is single-threaded by default).
_JUDGE_CONCURRENCY = 5

_JUDGE_SYSTEM = (
    "You are an objective evaluator assessing whether an AI agent's answer "
    "correctly addresses a business question.\n\n"
    "Score the answer from 0.0 to 1.0:\n"
    "  1.0 — Fully correct: directly addresses the question, cites relevant data, "
    "meets all evaluation criteria\n"
    "  0.7 — Mostly correct: right direction, minor gaps in precision or completeness\n"
    "  0.4 — Partially correct: some relevant content but significant errors or omissions\n"
    "  0.0 — Incorrect, irrelevant, or empty\n\n"
    "Respond with ONLY a number between 0.0 and 1.0.  No explanation."
)


class AnswerJudge:
    """Scores workflow answers using an LLM-as-judge approach.

    Respects LLM_PROVIDER from config — works with Anthropic, OpenAI, and Ollama.
    When JUDGE_MODEL is set in config it uses a separate (typically lighter) model
    for scoring, keeping the main benchmark model available for workflow runs.

    Concurrency is capped at _JUDGE_CONCURRENCY to avoid overwhelming cloud-provider
    rate limits and local Ollama instances alike.
    """

    def __init__(self) -> None:
        judge_model = settings.judge_model or None  # None → use llm_model
        self._llm = LLMClient(model_override=judge_model)
        self._sem = asyncio.Semaphore(_JUDGE_CONCURRENCY)

    async def score(self, task: Task, answer: str | None) -> float | None:
        """Return a quality score 0.0–1.0, or None if the task has no ground truth.

        None means the task is excluded from avg_quality_score — it is not
        treated as a failure.  A score of 0.0 means the answer was evaluated
        and found to be wrong.

        Concurrency is bounded by the semaphore set at construction.
        """
        if not task.ground_truth and not task.evaluation_criteria:
            return None
        if not answer or not str(answer).strip():
            return 0.0

        prompt = (
            f"Task: {task.description}\n\n"
            f"Evaluation criteria: {task.evaluation_criteria}\n"
            f"Ground truth hint: {task.ground_truth}\n\n"
            f"Agent answer:\n{answer}\n\n"
            "Score (0.0–1.0):"
        )
        async with self._sem:
            try:
                text, _ = await self._llm.invoke(
                    [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]
                )
                return _parse_score(text)
            except Exception as exc:
                log.warning("Answer judge failed for task %s: %s", task.id, exc)
                return None


def _parse_score(text: str) -> float:
    """Extract a float score from LLM response text, clamped to [0.0, 1.0].

    Handles responses like "0.8", "0.80", "1", "1.0", ".7", "1.5" (clamped to 1.0).
    Falls back to 0.0 if no valid number is found.
    """
    match = re.search(r"(\d+(?:\.\d+)?|\.\d+)", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.0
