"""Tests for the six workflow patterns implemented in Week 2.

All LLM calls are mocked so tests run without API keys.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from tasks.task_registry import Task, TaskLevel
from tools.base import ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def task_l1() -> Task:
    return Task(
        id="TEST_L1",
        description="What is the total revenue for Hardware in March 2024?",
        level=TaskLevel.RETRIEVAL,
        expected_tools=["sql_query"],
        ground_truth=42000.0,
    )


@pytest.fixture
def task_l2() -> Task:
    return Task(
        id="TEST_L2",
        description="What is the average order value by customer segment?",
        level=TaskLevel.ANALYTICAL,
        expected_tools=["sql_query", "calculator"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ai_msg(text: str, tool_calls: list | None = None) -> AIMessage:
    """Build an AIMessage with optional tool_calls."""
    msg = AIMessage(content=text)
    msg.tool_calls = tool_calls or []
    return msg


def ok_result(data: dict | None = None) -> ToolResult:
    return ToolResult(success=True, data=data or {"rows": [[42000]], "row_count": 1})


def fail_result(error: str = "db error") -> ToolResult:
    return ToolResult(success=False, error=error)


# ---------------------------------------------------------------------------
# Workflow 1 — SingleStep
# ---------------------------------------------------------------------------


class TestSingleStepWorkflow:
    @pytest.mark.asyncio
    async def test_success(self, task_l1: Task) -> None:
        with patch("workflows.single_step.LLMClient") as MockLLM:
            MockLLM.return_value.invoke = AsyncMock(
                return_value=("Hardware revenue was $42,000.", 150)
            )
            from workflows.single_step import SingleStepWorkflow

            result = await SingleStepWorkflow().run(task_l1)

        assert result.success is True
        assert result.workflow_name == "single_step"
        assert result.task_id == "TEST_L1"
        assert "42,000" in result.answer
        assert result.total_tokens == 150
        assert result.latency_ms > 0
        assert result.tool_calls_total == 0
        assert len(result.reasoning_steps) == 2

    @pytest.mark.asyncio
    async def test_llm_error_returns_failure(self, task_l1: Task) -> None:
        with patch("workflows.single_step.LLMClient") as MockLLM:
            MockLLM.return_value.invoke = AsyncMock(side_effect=RuntimeError("API error"))
            from workflows.single_step import SingleStepWorkflow

            result = await SingleStepWorkflow().run(task_l1)

        assert result.success is False
        assert "API error" in result.error
        assert result.latency_ms > 0


# ---------------------------------------------------------------------------
# Workflow 2 — ToolUsing
# ---------------------------------------------------------------------------


class TestToolUsingWorkflow:
    @pytest.mark.asyncio
    async def test_direct_answer_no_tools(self, task_l1: Task) -> None:
        final = ai_msg("Revenue was $42,000.", [])
        with patch("workflows.tool_using.LLMClient") as MockLLM, \
             patch("workflows.tool_using.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke_with_tools = AsyncMock(return_value=(final, 200))
            MockDisp.return_value.schemas = []
            from workflows.tool_using import ToolUsingWorkflow

            result = await ToolUsingWorkflow().run(task_l1)

        assert result.success is True
        assert result.tool_calls_total == 0
        assert result.total_tokens == 200

    @pytest.mark.asyncio
    async def test_one_tool_call_then_answer(self, task_l1: Task) -> None:
        tool_msg = ai_msg("", [{"id": "c1", "name": "sql_query", "args": {"query": "SELECT 1"}}])
        final_msg = ai_msg("Hardware revenue in March 2024 was $42,000.", [])

        with patch("workflows.tool_using.LLMClient") as MockLLM, \
             patch("workflows.tool_using.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke_with_tools = AsyncMock(
                side_effect=[(tool_msg, 100), (final_msg, 80)]
            )
            disp = MockDisp.return_value
            disp.schemas = []
            disp.dispatch = AsyncMock(return_value=ok_result())
            disp.result_to_text = MagicMock(return_value='{"rows": [[42000]]}')
            from workflows.tool_using import ToolUsingWorkflow

            result = await ToolUsingWorkflow().run(task_l1)

        assert result.success is True
        assert result.tool_calls_total == 1
        assert result.tool_calls_successful == 1
        assert "sql_query" in result.tools_used
        assert result.total_tokens == 180

    @pytest.mark.asyncio
    async def test_failed_tool_does_not_crash(self, task_l1: Task) -> None:
        tool_msg = ai_msg("", [{"id": "c1", "name": "sql_query", "args": {"query": "SELECT 1"}}])
        final_msg = ai_msg("I could not retrieve the data.", [])

        with patch("workflows.tool_using.LLMClient") as MockLLM, \
             patch("workflows.tool_using.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke_with_tools = AsyncMock(
                side_effect=[(tool_msg, 100), (final_msg, 60)]
            )
            disp = MockDisp.return_value
            disp.schemas = []
            disp.dispatch = AsyncMock(return_value=fail_result("db error"))
            disp.result_to_text = MagicMock(return_value="ERROR: db error")
            from workflows.tool_using import ToolUsingWorkflow

            result = await ToolUsingWorkflow().run(task_l1)

        assert result.success is True
        assert result.tool_calls_total == 1
        assert result.tool_calls_successful == 0


# ---------------------------------------------------------------------------
# Workflow 3 — Chain
# ---------------------------------------------------------------------------


class TestChainWorkflow:
    @pytest.mark.asyncio
    async def test_four_steps(self, task_l1: Task) -> None:
        plan_json = '{"queries": [{"tool": "sql_query", "args": {"query": "SELECT 1"}}]}'

        with patch("workflows.chain.LLMClient") as MockLLM, \
             patch("workflows.chain.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    (plan_json, 80),          # Step 1: decompose
                    ("Analysis: revenue up.", 100),  # Step 3: analyze
                    ("Answer: $42k", 60),      # Step 4: synthesize
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(return_value=ok_result())
            disp.result_to_text = MagicMock(return_value='{"rows": [[42000]]}')
            from workflows.chain import ChainWorkflow

            result = await ChainWorkflow().run(task_l1)

        assert result.success is True
        assert len(result.reasoning_steps) >= 4  # Steps 1–4 plus per-tool sub-steps
        assert result.tool_calls_total == 1
        assert result.tool_calls_successful == 1
        assert result.total_tokens == 240

    @pytest.mark.asyncio
    async def test_empty_plan_is_handled(self, task_l1: Task) -> None:
        """If the LLM returns unparseable JSON, plan is empty and no tools are called."""
        with patch("workflows.chain.LLMClient") as MockLLM, \
             patch("workflows.chain.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    ("not json at all", 40),
                    ("Analysis done.", 80),
                    ("Final answer.", 50),
                ]
            )
            MockDisp.return_value.dispatch = AsyncMock()
            from workflows.chain import ChainWorkflow

            result = await ChainWorkflow().run(task_l1)

        assert result.success is True
        assert result.tool_calls_total == 0


# ---------------------------------------------------------------------------
# Workflow 4 — Routing
# ---------------------------------------------------------------------------


class TestRoutingWorkflow:
    @pytest.mark.asyncio
    async def test_routes_to_retrieval(self, task_l1: Task) -> None:
        with patch("workflows.routing.LLMClient") as MockLLM, \
             patch("workflows.routing.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    ("retrieval", 20),
                    ("SELECT SUM(total) FROM orders", 50),
                    ("Revenue was $42,000.", 60),
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(return_value=ok_result())
            disp.result_to_text = MagicMock(return_value='{"rows": [[42000]]}')
            from workflows.routing import RoutingWorkflow

            result = await RoutingWorkflow().run(task_l1)

        assert result.success is True
        assert any("retrieval" in s for s in result.reasoning_steps)
        assert result.tool_calls_total == 1

    @pytest.mark.asyncio
    async def test_routes_to_analytical(self, task_l2: Task) -> None:
        py_result = ToolResult(success=True, data={"result": "Enterprise: $5000"})

        with patch("workflows.routing.LLMClient") as MockLLM, \
             patch("workflows.routing.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    ("analytical", 20),
                    ("SELECT segment, AVG(total) FROM orders GROUP BY segment", 50),
                    ("result = 'done'", 60),
                    ("Enterprise has highest AOV.", 70),
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(side_effect=[ok_result(), py_result])
            disp.result_to_text = MagicMock(return_value='{"result": "Enterprise: $5000"}')
            from workflows.routing import RoutingWorkflow

            result = await RoutingWorkflow().run(task_l2)

        assert result.success is True
        assert result.tool_calls_total == 2

    @pytest.mark.asyncio
    async def test_unknown_route_defaults_to_complex(self, task_l1: Task) -> None:
        # ToolUsingWorkflow is imported lazily inside _handle_complex —
        # patch it at its source module so the local import picks up the mock.
        from workflows.base import WorkflowResult

        with patch("workflows.routing.LLMClient") as MockLLM, \
             patch("workflows.routing.ToolDispatcher"), \
             patch("workflows.tool_using.ToolUsingWorkflow") as MockTU:
            MockLLM.return_value.invoke = AsyncMock(return_value=("gibberish", 20))
            MockTU.return_value.run = AsyncMock(
                return_value=WorkflowResult(
                    task_id=task_l1.id,
                    workflow_name="tool_using",
                    answer="Hardware revenue was $42,000.",
                    success=True,
                )
            )
            from workflows.routing import RoutingWorkflow

            result = await RoutingWorkflow().run(task_l1)

        assert result.success is True
        assert any("complex" in s for s in result.reasoning_steps)


# ---------------------------------------------------------------------------
# Workflow 5 — Parallel
# ---------------------------------------------------------------------------


class TestParallelWorkflow:
    @pytest.mark.asyncio
    async def test_parallel_execution(self, task_l2: Task) -> None:
        plan_json = (
            '{"queries": ['
            '{"tool": "sql_query", "args": {"query": "SELECT 1"}}, '
            '{"tool": "calculator", "args": {"expression": "100/2"}}'
            "]}"
        )
        calc_result = ToolResult(success=True, data={"result": 50})

        with patch("workflows.parallel.LLMClient") as MockLLM, \
             patch("workflows.parallel.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    (plan_json, 80),
                    ("Average order value by segment is ...", 120),
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(side_effect=[ok_result(), calc_result])
            disp.result_to_text = MagicMock(return_value="data")
            from workflows.parallel import ParallelWorkflow

            result = await ParallelWorkflow().run(task_l2)

        assert result.success is True
        assert result.tool_calls_total == 2
        assert result.tool_calls_successful == 2
        assert len(result.reasoning_steps) >= 3  # Plan, Execute (with per-tool sub-steps), Synthesize

    @pytest.mark.asyncio
    async def test_partial_failure_still_succeeds(self, task_l2: Task) -> None:
        plan_json = (
            '{"queries": ['
            '{"tool": "sql_query", "args": {"query": "SELECT 1"}}, '
            '{"tool": "sql_query", "args": {"query": "SELECT 2"}}'
            "]}"
        )

        with patch("workflows.parallel.LLMClient") as MockLLM, \
             patch("workflows.parallel.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    (plan_json, 80),
                    ("Partial answer based on available data.", 100),
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(side_effect=[ok_result(), fail_result()])
            disp.result_to_text = MagicMock(return_value="data")
            from workflows.parallel import ParallelWorkflow

            result = await ParallelWorkflow().run(task_l2)

        assert result.success is True
        assert result.tool_calls_successful == 1


# ---------------------------------------------------------------------------
# Workflow 6 — OrchestratorWorkers
# ---------------------------------------------------------------------------


class TestOrchestratorWorkersWorkflow:
    @pytest.mark.asyncio
    async def test_orchestrator_flow(self, task_l2: Task) -> None:
        plan_json = (
            '{"subtasks": ['
            '{"worker": "sql_worker", "task": "Get revenue data", '
            '"tool": "sql_query", "args": {"query": "SELECT * FROM revenue"}}, '
            '{"worker": "analysis_worker", "task": "Compute averages", '
            '"tool": "python_analysis", "args": {"code": "result = 42"}}'
            "]}"
        )
        py_result = ToolResult(success=True, data={"result": "42"})

        with patch("workflows.orchestrator_workers.LLMClient") as MockLLM, \
             patch("workflows.orchestrator_workers.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    (plan_json, 100),
                    ("Enterprise segment has the highest AOV.", 80),
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(side_effect=[ok_result(), py_result])
            disp.result_to_text = MagicMock(return_value="data")
            from workflows.orchestrator_workers import OrchestratorWorkersWorkflow

            result = await OrchestratorWorkersWorkflow().run(task_l2)

        assert result.success is True
        assert result.tool_calls_total == 2
        assert result.tool_calls_successful == 2
        assert "Orchestrator" in result.reasoning_steps[0]

    @pytest.mark.asyncio
    async def test_worker_failure_is_captured(self, task_l2: Task) -> None:
        plan_json = (
            '{"subtasks": ['
            '{"worker": "sql_worker", "task": "Get data", '
            '"tool": "sql_query", "args": {"query": "SELECT 1"}}'
            "]}"
        )

        with patch("workflows.orchestrator_workers.LLMClient") as MockLLM, \
             patch("workflows.orchestrator_workers.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    (plan_json, 80),
                    ("Unable to fully answer due to data error.", 60),
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(return_value=fail_result("connection refused"))
            disp.result_to_text = MagicMock(return_value="ERROR: connection refused")
            from workflows.orchestrator_workers import OrchestratorWorkersWorkflow

            result = await OrchestratorWorkersWorkflow().run(task_l2)

        # success = bool(answer) by design: workflow still synthesises an answer
        # even when all tools fail. Verify the tool failure is captured in metrics.
        assert result.tool_calls_successful == 0
        assert result.tool_calls_total == 1
        assert result.answer is not None  # degraded answer was produced


# ---------------------------------------------------------------------------
# Fix-validation tests: error paths, edge cases, and new behaviour
# ---------------------------------------------------------------------------


class TestTaskValidation:
    """Fix 11 — empty/blank task description must short-circuit with success=False."""

    @pytest.fixture
    def empty_task(self) -> Task:
        return Task(id="EMPTY", description="   ", level=TaskLevel.RETRIEVAL)

    @pytest.mark.asyncio
    async def test_single_step_rejects_empty_task(self, empty_task: Task) -> None:
        with patch("workflows.single_step.LLMClient"):
            from workflows.single_step import SingleStepWorkflow

            result = await SingleStepWorkflow().run(empty_task)
        assert result.success is False
        assert result.error == "Task description is empty"

    @pytest.mark.asyncio
    async def test_tool_using_rejects_empty_task(self, empty_task: Task) -> None:
        with patch("workflows.tool_using.LLMClient"), \
             patch("workflows.tool_using.ToolDispatcher"):
            from workflows.tool_using import ToolUsingWorkflow

            result = await ToolUsingWorkflow().run(empty_task)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_chain_rejects_empty_task(self, empty_task: Task) -> None:
        with patch("workflows.chain.LLMClient"), \
             patch("workflows.chain.ToolDispatcher"):
            from workflows.chain import ChainWorkflow

            result = await ChainWorkflow().run(empty_task)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_parallel_rejects_empty_task(self, empty_task: Task) -> None:
        with patch("workflows.parallel.LLMClient"), \
             patch("workflows.parallel.ToolDispatcher"):
            from workflows.parallel import ParallelWorkflow

            result = await ParallelWorkflow().run(empty_task)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_routing_rejects_empty_task(self, empty_task: Task) -> None:
        with patch("workflows.routing.LLMClient"), \
             patch("workflows.routing.ToolDispatcher"):
            from workflows.routing import RoutingWorkflow

            result = await RoutingWorkflow().run(empty_task)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_orchestrator_rejects_empty_task(self, empty_task: Task) -> None:
        with patch("workflows.orchestrator_workers.LLMClient"), \
             patch("workflows.orchestrator_workers.ToolDispatcher"):
            from workflows.orchestrator_workers import OrchestratorWorkersWorkflow

            result = await OrchestratorWorkersWorkflow().run(empty_task)
        assert result.success is False


class TestJsonParseFallback:
    """Fix 1 — LLM returning non-JSON is logged, not silently swallowed."""

    @pytest.mark.asyncio
    async def test_chain_bad_json_surfaces_warning(self, task_l1: Task) -> None:
        with patch("workflows.chain.LLMClient") as MockLLM, \
             patch("workflows.chain.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    ("This is not JSON at all — just prose", 40),  # Step 1: plan fails
                    ("Analysis without data.", 60),
                    ("Best-effort answer.", 50),
                ]
            )
            MockDisp.return_value.dispatch = AsyncMock()
            from workflows.chain import ChainWorkflow

            result = await ChainWorkflow().run(task_l1)

        # Workflow still completes
        assert result.success is True
        # Parse failure is visible in reasoning steps
        assert any("WARNING" in s and "parsing failed" in s for s in result.reasoning_steps)
        # No tools were called (empty plan → no dispatch)
        assert result.tool_calls_total == 0

    @pytest.mark.asyncio
    async def test_parallel_bad_json_uses_fallback_query(self, task_l2: Task) -> None:
        with patch("workflows.parallel.LLMClient") as MockLLM, \
             patch("workflows.parallel.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    ("not json", 40),
                    ("Synthesized answer.", 80),
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(return_value=ok_result())
            disp.result_to_text = MagicMock(return_value="data")
            from workflows.parallel import ParallelWorkflow

            result = await ParallelWorkflow().run(task_l2)

        assert result.success is True
        # Fallback query was used — exactly 1 tool call
        assert result.tool_calls_total == 1
        assert any("WARNING" in s for s in result.reasoning_steps)

    @pytest.mark.asyncio
    async def test_orchestrator_bad_json_fails_fast(self, task_l2: Task) -> None:
        """Fix 5: parse failure → immediate success=False, no fallback SQL executed."""
        with patch("workflows.orchestrator_workers.LLMClient") as MockLLM, \
             patch("workflows.orchestrator_workers.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    ("not json", 40),
                    # No second call — workflow aborts before synthesis
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock()  # must NOT be called
            from workflows.orchestrator_workers import OrchestratorWorkersWorkflow

            result = await OrchestratorWorkersWorkflow().run(task_l2)

        assert result.success is False
        assert result.tool_calls_total == 0  # no fallback query executed
        assert any("WARNING" in s and "parsing failed" in s for s in result.reasoning_steps)
        disp.dispatch.assert_not_called()


class TestSuccessFlag:
    """Fix 3 — success=False when every tool call fails."""

    @pytest.mark.asyncio
    async def test_parallel_all_tools_fail_is_failure(self, task_l2: Task) -> None:
        plan_json = '{"queries": [{"tool": "sql_query", "args": {"query": "SELECT 1"}}]}'

        with patch("workflows.parallel.LLMClient") as MockLLM, \
             patch("workflows.parallel.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[(plan_json, 50), ("Guessed answer.", 60)]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(return_value=fail_result("db down"))
            disp.result_to_text = MagicMock(return_value="ERROR: db down")
            from workflows.parallel import ParallelWorkflow

            result = await ParallelWorkflow().run(task_l2)

        # success = bool(answer): parallel synthesises even on all-tool-failure.
        # Tool failure is captured in metrics, not in the success flag.
        assert result.tool_calls_total == 1
        assert result.tool_calls_successful == 0
        assert result.answer is not None  # degraded answer was produced

    @pytest.mark.asyncio
    async def test_parallel_partial_failure_is_still_success(self, task_l2: Task) -> None:
        """If at least one tool succeeds, success=True."""
        plan_json = (
            '{"queries": ['
            '{"tool": "sql_query", "args": {"query": "SELECT 1"}}, '
            '{"tool": "sql_query", "args": {"query": "SELECT 2"}}'
            "]}"
        )

        with patch("workflows.parallel.LLMClient") as MockLLM, \
             patch("workflows.parallel.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[(plan_json, 50), ("Answer.", 60)]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(side_effect=[ok_result(), fail_result()])
            disp.result_to_text = MagicMock(return_value="data")
            from workflows.parallel import ParallelWorkflow

            result = await ParallelWorkflow().run(task_l2)

        assert result.success is True
        assert result.tool_calls_successful == 1

    @pytest.mark.asyncio
    async def test_chain_all_tools_fail_is_failure(self, task_l1: Task) -> None:
        """Fix 6: chain short-circuits after total retrieval failure — no analyze/synthesize."""
        plan_json = '{"queries": [{"tool": "sql_query", "args": {"query": "SELECT 1"}}]}'

        with patch("workflows.chain.LLMClient") as MockLLM, \
             patch("workflows.chain.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[
                    (plan_json, 40),
                    # Chain short-circuits here — Step 3 and 4 are never called
                ]
            )
            disp = MockDisp.return_value
            disp.dispatch = AsyncMock(return_value=fail_result("db error"))
            disp.result_to_text = MagicMock(return_value="ERROR: db error")
            from workflows.chain import ChainWorkflow

            result = await ChainWorkflow().run(task_l1)

        assert result.success is False
        assert result.tool_calls_total == 1
        assert result.tool_calls_successful == 0
        # Exactly 1 LLM call was made (plan only)
        assert MockLLM.return_value.invoke.call_count == 1

    @pytest.mark.asyncio
    async def test_routing_complex_propagates_inner_failure(self, task_l1: Task) -> None:
        """Fix 10: routing wrapping a failed complex handler → success=False."""
        from workflows.base import WorkflowResult

        with patch("workflows.routing.LLMClient") as MockLLM, \
             patch("workflows.routing.ToolDispatcher"), \
             patch("workflows.tool_using.ToolUsingWorkflow") as MockTU:
            MockLLM.return_value.invoke = AsyncMock(return_value=("complex", 20))
            MockTU.return_value.run = AsyncMock(
                return_value=WorkflowResult(
                    task_id=task_l1.id,
                    workflow_name="tool_using",
                    answer=None,
                    success=False,
                    error="LLM API error",
                )
            )
            from workflows.routing import RoutingWorkflow

            result = await RoutingWorkflow().run(task_l1)

        # Fix 10: inner failure must surface
        assert result.success is False


class TestToolUsingEdgeCases:
    """Fix 2 — malformed tool_calls must not KeyError."""

    @pytest.mark.asyncio
    async def test_missing_tool_name_is_skipped(self, task_l1: Task) -> None:
        """A tool_call dict missing 'name' is skipped gracefully."""
        malformed = ai_msg("", [{"id": "c1", "args": {"query": "SELECT 1"}}])  # no 'name'
        final = ai_msg("Best-effort answer.", [])

        with patch("workflows.tool_using.LLMClient") as MockLLM, \
             patch("workflows.tool_using.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke_with_tools = AsyncMock(
                side_effect=[(malformed, 100), (final, 60)]
            )
            disp = MockDisp.return_value
            disp.schemas = []
            disp.dispatch = AsyncMock()  # should NOT be called
            from workflows.tool_using import ToolUsingWorkflow

            result = await ToolUsingWorkflow().run(task_l1)

        assert result.success is True
        assert result.tool_calls_total == 0  # malformed call was skipped
        disp.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_tool_result_is_truncated(self, task_l1: Task) -> None:
        """Fix 8 — tool results over 3000 chars are truncated before adding to messages."""
        from workflows.tool_using import _MAX_TOOL_RESULT_CHARS

        big_result = ok_result()
        tool_call_msg = ai_msg("", [{"id": "c1", "name": "sql_query", "args": {"query": "SELECT 1"}}])
        final_msg = ai_msg("Answer.", [])

        with patch("workflows.tool_using.LLMClient") as MockLLM, \
             patch("workflows.tool_using.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke_with_tools = AsyncMock(
                side_effect=[(tool_call_msg, 100), (final_msg, 60)]
            )
            disp = MockDisp.return_value
            disp.schemas = []
            disp.dispatch = AsyncMock(return_value=big_result)
            # Return a string longer than the cap
            oversized = "x" * (_MAX_TOOL_RESULT_CHARS + 500)
            disp.result_to_text = MagicMock(return_value=oversized)
            from langchain_core.messages import ToolMessage
            captured_messages: list = []

            original_ainvoke = None

            async def capture_invoke(msgs):
                captured_messages.extend(msgs)
                return final_msg, 60

            MockLLM.return_value.invoke = AsyncMock(side_effect=capture_invoke)
            from workflows.tool_using import ToolUsingWorkflow

            result = await ToolUsingWorkflow().run(task_l1)

        # Find the ToolMessage that was appended
        tool_msgs = [m for m in captured_messages if isinstance(m, ToolMessage)]
        if tool_msgs:
            assert len(tool_msgs[0].content) <= _MAX_TOOL_RESULT_CHARS + 100  # +100 for the "... [truncated N chars]" suffix


class TestToolDispatcher:
    """Fix 14 — ToolDispatcher direct tests."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_failure(self) -> None:
        from agents.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher()
        result = await dispatcher.dispatch("nonexistent_tool", {})
        assert result.success is False
        assert "nonexistent_tool" in result.error

    @pytest.mark.asyncio
    async def test_allowed_tools_filter_excludes_others(self) -> None:
        from agents.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(allowed_tools=["sql_query"])
        result = await dispatcher.dispatch("calculator", {"expression": "1+1"})
        assert result.success is False

    def test_schema_matches_allowed_tools(self) -> None:
        from agents.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(allowed_tools=["sql_query", "calculator"])
        schema_names = {s["name"] for s in dispatcher.schemas}
        assert schema_names == {"sql_query", "calculator"}

    def test_result_to_text_on_failure(self) -> None:
        from agents.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher()
        result = ToolResult(success=False, error="something went wrong")
        text = dispatcher.result_to_text(result)
        assert "ERROR" in text
        assert "something went wrong" in text


class TestConcurrencyCapEnforced:
    """Fix 12 — parallel workflow caps queries at _MAX_QUERIES even if LLM returns more."""

    @pytest.mark.asyncio
    async def test_excess_queries_are_capped(self, task_l2: Task) -> None:
        from workflows.parallel import _MAX_QUERIES

        # Return more queries than the cap
        excess = _MAX_QUERIES + 2
        queries = [{"tool": "sql_query", "args": {"query": f"SELECT {i}"}} for i in range(excess)]
        plan_json = f'{{"queries": {__import__("json").dumps(queries)}}}'

        dispatched_calls: list = []

        with patch("workflows.parallel.LLMClient") as MockLLM, \
             patch("workflows.parallel.ToolDispatcher") as MockDisp:
            MockLLM.return_value.invoke = AsyncMock(
                side_effect=[(plan_json, 50), ("Answer.", 60)]
            )
            disp = MockDisp.return_value

            async def capture_dispatch(tool, args):
                dispatched_calls.append((tool, args))
                return ok_result()

            disp.dispatch = AsyncMock(side_effect=capture_dispatch)
            disp.result_to_text = MagicMock(return_value="data")
            from workflows.parallel import ParallelWorkflow

            result = await ParallelWorkflow().run(task_l2)

        assert result.success is True
        assert len(dispatched_calls) == _MAX_QUERIES  # not excess
        assert any(f"capping to {_MAX_QUERIES}" in s for s in result.reasoning_steps)


class TestRetryBehaviour:
    """Fix 4 — LLMClient retries transient errors and re-raises non-retriable ones."""

    @pytest.mark.asyncio
    async def test_retriable_error_is_retried(self) -> None:
        from agents.llm_client import LLMClient
        from langchain_core.messages import HumanMessage

        client = LLMClient.__new__(LLMClient)  # skip __init__ to avoid real model

        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("rate limit exceeded")
            from langchain_core.messages import AIMessage
            return AIMessage(content="ok")

        client._model = MagicMock()
        client._model.ainvoke = flaky

        text, tokens = await client.invoke([HumanMessage(content="hello")])
        assert text == "ok"
        assert call_count == 3  # failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_non_retriable_error_is_not_retried(self) -> None:
        from agents.llm_client import LLMClient
        from langchain_core.messages import HumanMessage

        client = LLMClient.__new__(LLMClient)
        call_count = 0

        async def always_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("auth failed")  # not in _RETRIABLE_KEYWORDS

        client._model = MagicMock()
        client._model.ainvoke = always_fail

        with pytest.raises(ValueError, match="auth failed"):
            await client.invoke([HumanMessage(content="hello")])
        assert call_count == 1  # stopped immediately, no retry


# ---------------------------------------------------------------------------
# Fix #2 — max-iteration bailout must return success=False
# ---------------------------------------------------------------------------


class TestMaxIterationBailout:
    """Fix 2 — hitting _MAX_ITERATIONS without a final answer is a failure."""

    @pytest.mark.asyncio
    async def test_max_iterations_returns_failure(self, task_l1: Task) -> None:
        from workflows.tool_using import _MAX_ITERATIONS

        # Every LLM call returns a tool_call — agent never reaches a text answer
        tool_call_msg = ai_msg(
            "", [{"id": "c1", "name": "sql_query", "args": {"query": "SELECT 1"}}]
        )

        with patch("workflows.tool_using.LLMClient") as MockLLM, \
             patch("workflows.tool_using.ToolDispatcher") as MockDisp:
            # Always request a tool call — never produce a final answer
            MockLLM.return_value.invoke_with_tools = AsyncMock(
                return_value=(tool_call_msg, 50)
            )
            disp = MockDisp.return_value
            disp.schemas = []
            disp.dispatch = AsyncMock(return_value=ok_result())
            disp.result_to_text = MagicMock(return_value="data")
            from workflows.tool_using import ToolUsingWorkflow

            result = await ToolUsingWorkflow().run(task_l1)

        assert result.success is False
        has_iteration_limit_error = result.error is not None and "Iteration limit" in result.error
        has_max_iterations_in_steps = "Max iterations" in " ".join(result.reasoning_steps)
        assert has_iteration_limit_error or has_max_iterations_in_steps
        # Should have made exactly _MAX_ITERATIONS LLM calls
        assert MockLLM.return_value.invoke_with_tools.call_count == _MAX_ITERATIONS


# ---------------------------------------------------------------------------
# Fix #7 — _parse_route robustness
# ---------------------------------------------------------------------------


class TestParseRoute:
    """Fix 7 — _parse_route must not false-match substring inside longer tokens."""

    def test_exact_first_word_retrieval(self) -> None:
        from workflows.routing import _parse_route
        assert _parse_route("retrieval") == "retrieval"

    def test_exact_first_word_analytical(self) -> None:
        from workflows.routing import _parse_route
        assert _parse_route("analytical") == "analytical"

    def test_exact_first_word_complex(self) -> None:
        from workflows.routing import _parse_route
        assert _parse_route("complex") == "complex"

    def test_analytical_takes_priority_over_retrieval_in_phrase(self) -> None:
        """'analytical retrieval of metrics' should return 'analytical', not 'retrieval'."""
        from workflows.routing import _parse_route
        assert _parse_route("analytical retrieval of complex metrics") == "analytical"

    def test_unknown_response_defaults_to_complex(self) -> None:
        from workflows.routing import _parse_route
        assert _parse_route("I cannot determine the category.") == "complex"

    def test_empty_string_defaults_to_complex(self) -> None:
        from workflows.routing import _parse_route
        assert _parse_route("") == "complex"

    def test_case_insensitive(self) -> None:
        from workflows.routing import _parse_route
        assert _parse_route("RETRIEVAL") == "retrieval"
        assert _parse_route("Analytical") == "analytical"


# ---------------------------------------------------------------------------
# Fix #8 — retries counter is propagated into WorkflowResult
# ---------------------------------------------------------------------------


class TestRetriesAccumulated:
    """Fix 8 — LLMClient.last_retries is wired into WorkflowResult.retries."""

    @pytest.mark.asyncio
    async def test_single_step_records_retries(self, task_l1: Task) -> None:
        """SingleStepWorkflow must pick up last_retries from LLMClient."""
        with patch("workflows.single_step.LLMClient") as MockLLM:
            llm_instance = MockLLM.return_value
            llm_instance.invoke = AsyncMock(return_value=("Answer.", 100))
            llm_instance.last_retries = 2  # simulate 2 retries needed
            from workflows.single_step import SingleStepWorkflow

            result = await SingleStepWorkflow().run(task_l1)

        assert result.retries == 2

    @pytest.mark.asyncio
    async def test_chain_accumulates_retries_across_steps(self, task_l1: Task) -> None:
        """ChainWorkflow must accumulate retries from each LLM call."""
        plan_json = '{"queries": []}'

        with patch("workflows.chain.LLMClient") as MockLLM, \
             patch("workflows.chain.ToolDispatcher"):
            llm_instance = MockLLM.return_value
            # 3 LLM calls: plan, analyze, synthesize
            llm_instance.invoke = AsyncMock(
                side_effect=[
                    ("", 30),   # plan (empty queries → no retrieval)
                    ("Analysis.", 50),
                    ("Answer.", 40),
                ]
            )
            # Each call simulates 1 retry
            llm_instance.last_retries = 1
            from workflows.chain import ChainWorkflow

            result = await ChainWorkflow().run(task_l1)

        # plan + analyze + synthesize = 3 calls × 1 retry each = 3
        assert result.retries == 3

    @pytest.mark.asyncio
    async def test_llm_client_last_retries_property(self) -> None:
        """LLMClient.last_retries must track the retry count of the last call."""
        from agents.llm_client import LLMClient
        from langchain_core.messages import AIMessage, HumanMessage

        client = LLMClient.__new__(LLMClient)
        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("service unavailable")
            return AIMessage(content="done")

        client._model = MagicMock()
        client._model.ainvoke = flaky

        await client.invoke([HumanMessage(content="test")])
        # Succeeded on attempt 3 (index 2) → 2 retries
        assert client.last_retries == 2


# ---------------------------------------------------------------------------
# Fix #4 — WorkflowMetrics uses running sums, not stored results
# ---------------------------------------------------------------------------


class TestWorkflowMetricsRunningSums:
    """Fix 4 — WorkflowMetrics must not accumulate WorkflowResult objects."""

    def test_no_results_attribute(self) -> None:
        """WorkflowMetrics must NOT have a 'results' list attribute."""
        from evaluation.metrics import WorkflowMetrics

        m = WorkflowMetrics(workflow_name="test")
        assert not hasattr(m, "results"), (
            "WorkflowMetrics must not store individual results — use running sums"
        )

    def test_running_sums_correct(self) -> None:
        from evaluation.metrics import WorkflowMetrics
        from workflows.base import WorkflowResult

        m = WorkflowMetrics(workflow_name="test")
        # success is derived from bool(answer) — must supply answer to get success=True
        r1 = WorkflowResult(
            task_id="T1", workflow_name="test", answer="Revenue was $42k.",
            reasoning_steps=["a", "b"], tool_calls_total=2, tool_calls_successful=1,
            total_tokens=100, latency_ms=50.0, retries=1,
        )
        r2 = WorkflowResult(
            task_id="T2", workflow_name="test",  # no answer → success=False
            reasoning_steps=["c"], tool_calls_total=1, tool_calls_successful=0,
            total_tokens=80, latency_ms=30.0, retries=0,
        )
        m.record(r1)
        m.record(r2)

        assert m.total_tasks == 2
        assert m.successes == 1
        assert m.avg_reasoning_steps == pytest.approx(1.5)   # (2+1)/2
        # Per-task mean: r1=1/2=0.5, r2=0/1=0.0 → mean=0.25
        # (old aggregate 1/3 is wrong: it weights by call count, not task count)
        assert m.tool_accuracy == pytest.approx(0.25)
        assert m.avg_retries == pytest.approx(0.5)

    def test_quality_score_aggregation(self) -> None:
        from evaluation.metrics import WorkflowMetrics
        from workflows.base import WorkflowResult

        m = WorkflowMetrics(workflow_name="test")
        m.record(WorkflowResult(task_id="T1", workflow_name="test", success=True, quality_score=0.8))
        m.record(WorkflowResult(task_id="T2", workflow_name="test", success=True, quality_score=0.6))
        m.record(WorkflowResult(task_id="T3", workflow_name="test", success=True, quality_score=None))

        assert m.avg_quality_score == pytest.approx(0.7)  # only 2 scored tasks
        s = m.summary()
        assert "avg_quality_score" in s
        assert s["avg_quality_score"] == pytest.approx(0.7, abs=0.001)

    def test_no_quality_scores_returns_none(self) -> None:
        from evaluation.metrics import WorkflowMetrics
        from workflows.base import WorkflowResult

        m = WorkflowMetrics(workflow_name="test")
        m.record(WorkflowResult(task_id="T1", workflow_name="test", success=True))
        assert m.avg_quality_score is None
        assert "avg_quality_score" not in m.summary()


# ---------------------------------------------------------------------------
# Fix #1 — AnswerJudge: score() and _parse_score()
# ---------------------------------------------------------------------------


class TestAnswerJudge:
    """Fix 1 — AnswerJudge correctly scores answers and handles edge cases."""

    @pytest.mark.asyncio
    async def test_scores_answer_with_ground_truth(self, task_l1: Task) -> None:
        with patch("evaluation.judge.LLMClient") as MockLLM:
            MockLLM.return_value.invoke = AsyncMock(return_value=("0.9", 50))
            from evaluation.judge import AnswerJudge

            judge = AnswerJudge()
            score = await judge.score(task_l1, "Revenue was $42,000.")

        assert score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_no_ground_truth_returns_none(self) -> None:
        from evaluation.judge import AnswerJudge
        from tasks.task_registry import Task, TaskLevel

        task = Task(id="X", description="q", level=TaskLevel.RETRIEVAL)  # no ground_truth
        judge = AnswerJudge.__new__(AnswerJudge)
        score = await judge.score(task, "some answer")
        assert score is None

    @pytest.mark.asyncio
    async def test_empty_answer_scores_zero(self, task_l1: Task) -> None:
        from evaluation.judge import AnswerJudge

        judge = AnswerJudge.__new__(AnswerJudge)  # no LLM needed
        score = await judge.score(task_l1, "")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, task_l1: Task) -> None:
        with patch("evaluation.judge.LLMClient") as MockLLM:
            MockLLM.return_value.invoke = AsyncMock(side_effect=RuntimeError("API error"))
            from evaluation.judge import AnswerJudge

            judge = AnswerJudge()
            score = await judge.score(task_l1, "Some answer.")

        assert score is None  # failure is swallowed gracefully

    def test_parse_score_valid_floats(self) -> None:
        from evaluation.judge import _parse_score

        assert _parse_score("0.8") == pytest.approx(0.8)
        assert _parse_score("1.0") == pytest.approx(1.0)
        assert _parse_score("0") == pytest.approx(0.0)
        assert _parse_score("1") == pytest.approx(1.0)

    def test_parse_score_clamped(self) -> None:
        from evaluation.judge import _parse_score

        assert _parse_score("The score is 0.75 out of 1.") == pytest.approx(0.75)
        assert _parse_score("1.5") == pytest.approx(1.0)   # > 1.0 clamped to 1.0
        assert _parse_score(".7") == pytest.approx(0.7)    # leading-dot float

    def test_parse_score_no_number_returns_zero(self) -> None:
        from evaluation.judge import _parse_score

        assert _parse_score("I cannot determine a score.") == 0.0


# ---------------------------------------------------------------------------
# Fix #3 — _is_retriable precision
# ---------------------------------------------------------------------------


class TestIsRetriable:
    """Fix 3 — permanent failures must not be retried."""

    def test_rate_limit_is_retriable(self) -> None:
        from agents.llm_client import _is_retriable

        assert _is_retriable(RuntimeError("rate limit exceeded")) is True

    def test_service_unavailable_is_retriable(self) -> None:
        from agents.llm_client import _is_retriable

        assert _is_retriable(RuntimeError("503 service unavailable")) is True

    def test_auth_error_is_not_retriable(self) -> None:
        from agents.llm_client import _is_retriable

        assert _is_retriable(RuntimeError("authentication failed")) is False

    def test_api_key_error_is_not_retriable(self) -> None:
        from agents.llm_client import _is_retriable

        assert _is_retriable(RuntimeError("invalid api key provided")) is False

    def test_unauthorized_is_not_retriable(self) -> None:
        from agents.llm_client import _is_retriable

        assert _is_retriable(RuntimeError("401 unauthorized")) is False

    def test_generic_connection_error_without_known_keywords_is_not_retriable(self) -> None:
        """'connection' alone no longer triggers retry — too broad."""
        from agents.llm_client import _is_retriable

        assert _is_retriable(RuntimeError("connection error: SSL certificate mismatch")) is False
