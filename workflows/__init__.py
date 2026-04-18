"""Workflow registry — always produces fresh instances to avoid shared state."""

from __future__ import annotations

from workflows.base import BaseWorkflow, WorkflowResult
from workflows.chain import ChainWorkflow
from workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
from workflows.human_in_loop import HumanInLoopWorkflow
from workflows.manager_agent import ManagerAgentWorkflow
from workflows.multi_agent_handoff import MultiAgentHandoffWorkflow
from workflows.orchestrator_workers import OrchestratorWorkersWorkflow
from workflows.parallel import ParallelWorkflow
from workflows.routing import RoutingWorkflow
from workflows.single_step import SingleStepWorkflow
from workflows.tool_using import ToolUsingWorkflow


def get_all_workflows() -> list[BaseWorkflow]:
    """Return fresh workflow instances for a benchmark run.

    Always constructs new instances so each run starts with clean state.
    Using module-level singletons would cause state leakage between
    concurrent task executions.
    """
    return [
        SingleStepWorkflow(),
        ToolUsingWorkflow(),
        ChainWorkflow(),
        RoutingWorkflow(),
        ParallelWorkflow(),
        OrchestratorWorkersWorkflow(),
        EvaluatorOptimizerWorkflow(),
        ManagerAgentWorkflow(),
        MultiAgentHandoffWorkflow(),
        HumanInLoopWorkflow(),
    ]


def get_workflow_map() -> dict[str, BaseWorkflow]:
    """Return a name → workflow mapping with fresh instances."""
    return {w.name: w for w in get_all_workflows()}


__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "SingleStepWorkflow",
    "ToolUsingWorkflow",
    "ChainWorkflow",
    "RoutingWorkflow",
    "ParallelWorkflow",
    "OrchestratorWorkersWorkflow",
    "EvaluatorOptimizerWorkflow",
    "ManagerAgentWorkflow",
    "MultiAgentHandoffWorkflow",
    "HumanInLoopWorkflow",
    "get_all_workflows",
    "get_workflow_map",
]
