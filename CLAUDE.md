# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a benchmarking project that implements and compares 10 AI agent workflow patterns on a shared structured business dataset. Every workflow pattern runs the same tasks against the same data using the same LLM so results are directly comparable.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then fill in API keys

# Run benchmarks
python -m scripts.cli run                        # all workflows, all tasks
python -m scripts.cli run --workflow chain        # single workflow
python -m scripts.cli run --level 2               # tasks at a specific level
python -m scripts.cli run --task-id L1_T01         # single task
python -m scripts.cli list-workflows              # show available workflows
python -m scripts.cli list-tasks                  # show registered tasks

# Tests
pytest                    # all tests
pytest tests/test_foo.py  # single file
pytest -k "test_name"     # single test by name

# Lint & format
ruff check .              # lint
ruff check --fix .        # auto-fix
ruff format .             # format
mypy .                    # type check
```

## Architecture

### Key base classes (implement these when adding new workflows/tools)
- `tools/base.py` — `BaseTool` and `ToolResult`: every tool implements `async execute(**kwargs) -> ToolResult`
- `workflows/base.py` — `BaseWorkflow` and `WorkflowResult`: every workflow pattern implements `async run(task) -> WorkflowResult`
- `agents/llm_client.py` — `LLMClient`: shared LLM invocation layer used by all workflow patterns
- `tasks/task_registry.py` — `Task`, `TaskLevel`, and `TaskRegistry`: central task registry, tasks have levels 1–4
- `evaluation/metrics.py` — `WorkflowMetrics`: aggregates `WorkflowResult` objects into per-workflow stats
- `config/settings.py` — `Settings` singleton loaded from `.env`, accessed via `from config import settings`

### Data flow
1. Tasks are loaded from `tasks/` into the `TaskRegistry`
2. Each workflow pattern in `workflows/` runs every task via its `run()` method
3. Workflows use shared tools from `tools/` to interact with SQL, CSVs, vector DB
4. Results (`WorkflowResult`) are collected by `evaluation/metrics.py`
5. Output is written to `results/` and visualized in `plots/`

### Project structure
```
config/         — Settings (Pydantic, loaded from .env)
tools/          — Shared tools: SQL, CSV reader, vector search, calculator, doc retrieval
tasks/          — Task definitions (L1–L4) and registry
agents/         — Agent implementations used by workflow patterns
workflows/      — The 10 workflow pattern implementations
evaluation/     — Metrics aggregation and DeepEval integration
scripts/        — CLI (Typer) and utility scripts
data/csv/       — Source CSV files
data/db/        — SQLite database (gitignored)
data/vectordb/  — ChromaDB persistence (gitignored)
tests/          — Pytest test suite
results/        — Benchmark output (gitignored)
plots/          — Generated charts (gitignored)
```

## Tech Stack

- **Orchestration:** LangGraph or CrewAI
- **LLM:** Claude (preferred) or GPT-4 — must be swappable via a single config
- **Data:** SQLite (dev) / PostgreSQL (prod) + vector DB for business rules/docs
- **Evaluation:** DeepEval or LangSmith

## The 10 Workflow Patterns

Implemented in `agents/` and `workflows/`, one module per pattern:
1. Single-step LLM
2. Tool-using agent
3. Chain workflow
4. Routing workflow
5. Parallel workflow
6. Orchestrator-workers
7. Evaluator-optimizer
8. Manager agent
9. Multi-agent handoff
10. Human-in-the-loop workflow

## Task Levels

Defined in `tasks/`, used to drive all workflow benchmarks:
- **L1** — Retrieval (e.g., revenue lookup, product search)
- **L2** — Analytical (e.g., trends, segmentation)
- **L3** — Multi-step reasoning (e.g., performance explanation)
- **L4** — Decision-making (e.g., strategy recommendations)

## Evaluation Metrics

Every workflow run is evaluated on: task success rate, tool accuracy, reasoning steps, latency, token cost, retry count, and failure rate.

## Implementation Phases

- **Week 1:** Dataset ingestion, SQL/vector DB setup, shared tool layer
- **Week 2:** Workflows 1–6 implementation
- **Week 3:** Workflows 7–10, evaluation harness integration
- **Week 4:** Comparative analysis infrastructure: plots, analysis report, blog article generator
- **Week 5:** Run benchmarks, generate results, produce final plots/analysis/blog output
