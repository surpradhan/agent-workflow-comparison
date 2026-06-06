# Agent Workflow Benchmark

A controlled benchmark comparing **10 AI agent workflow patterns** on a shared structured business dataset. Every pattern runs the same 25 tasks against the same data using the same LLM — results are directly comparable.

## What it measures

Each workflow is evaluated on:

| Metric | What it measures |
|--------|-----------------|
| Success rate | Tasks that produced a non-empty answer |
| Quality score | LLM-as-judge rating (0–1) against ground truth |
| Tool accuracy | Fraction of tool calls that succeeded |
| Latency | Average wall-clock time per task |
| Tokens | Average total tokens per task |

## The 10 Workflow Patterns

| # | Pattern | Description |
|---|---------|-------------|
| 1 | Single-step | One LLM call. No tools. The baseline. |
| 2 | Tool-using | ReAct loop: model selects and calls tools iteratively. |
| 3 | Chain | Four deterministic stages: decompose → retrieve → analyse → synthesise. |
| 4 | Routing | A classifier decides which specialist handler to invoke. |
| 5 | Parallel | Multiple independent sub-queries run concurrently; results merged. |
| 6 | Orchestrator-workers | Central orchestrator fans out to worker agents in parallel. |
| 7 | Evaluator-optimizer | Generate → Evaluate → Optimize self-refinement loop. |
| 8 | Manager agent | Manager supervises specialist sub-agents sequentially. |
| 9 | Multi-agent handoff | Retrieval → Analysis → Synthesis, each a specialist agent. |
| 10 | Human-in-the-loop | Checkpoints pause for human approval before continuing. |

## Task Levels

| Level | Type | Example |
|-------|------|---------|
| L1 | Retrieval | "What is total revenue for Hardware in March 2024?" |
| L2 | Analytical | "What is the month-over-month revenue trend for 2024?" |
| L3 | Reasoning | "Why did the payment success rate drop in March 2024?" |
| L4 | Decision | "Should we raise prices on high-margin products or focus on volume?" |

## Key Results

Benchmark run with `llama3.1:8b` via Ollama on a local M1 Pro 16 GB:

| Rank | Workflow | Success | Quality | Latency (ms) |
|------|----------|---------|---------|--------------|
| 🥇 | `tool_using` | 100% | 0.572 | 13,581 |
| 🥈 | `parallel` | 100% | 0.696 | 30,362 |
| 🥉 | `single_step` | 100% | 0.748 | 23,089 |
| 4 | `routing` | 100% | 0.484 | 24,310 |
| 5 | `evaluator_optimizer` | 100% | 0.720 | 93,872 |
| 6 | `multi_agent_handoff` | 84% | 0.729 | 48,107 |
| 7 | `manager_agent` | 80% | 0.550 | 36,946 |
| 8 | `orchestrator_workers` | 64% | 0.637 | 25,417 |
| 9 | `chain` | 56% | 0.700 | 52,137 |
| 10 | `human_in_loop` | 44% | **0.836** | 23,978 |

> `human_in_loop` has the lowest completion rate but the highest answer quality — human checkpoints filter out poor answers before they're returned.

## Setup

```bash
# 1. Clone and create a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure your LLM provider
cp .env.example .env
# Edit .env and set your API key and preferred model

# 3. Initialise the dataset
python -m scripts.cli setup
```

### LLM Providers

| Provider | `.env` settings | Notes |
|----------|-----------------|-------|
| **Ollama** (local) | `LLM_PROVIDER=ollama`<br>`LLM_MODEL=llama3.1:8b` | Run `ollama pull llama3.1:8b` first. Recommended. |
| **Anthropic** | `LLM_PROVIDER=anthropic`<br>`LLM_MODEL=claude-sonnet-4-20250514` | Set `ANTHROPIC_API_KEY` |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`LLM_MODEL=gpt-4o` | Set `OPENAI_API_KEY` |
| **Groq** | `LLM_PROVIDER=groq`<br>`LLM_MODEL=llama-3.3-70b-versatile` | Set `GROQ_API_KEY`. Dev Tier recommended for full runs. |

## Running the Benchmark

```bash
# Full benchmark — all 10 workflows × 25 tasks
python -m scripts.cli run

# Single workflow
python -m scripts.cli run --workflow tool_using

# Single task level
python -m scripts.cli run --level 2

# Single task
python -m scripts.cli run --task-id L1_T01

# Generate plots, analysis report, and blog post from results
python -m scripts.cli plot
python -m scripts.cli analyze
python -m scripts.cli blog

# List available workflows and tasks
python -m scripts.cli list-workflows
python -m scripts.cli list-tasks
```

Results are written to `results/` as JSON. Plots are written to `plots/` as PNG.

## Development

```bash
# Run tests
pytest

# Lint and format
ruff check .
ruff format .

# Type check
mypy .
```

## Project Structure

```
config/          — Settings (Pydantic, loaded from .env)
tools/           — Shared tools: SQL, CSV reader, vector search, calculator
tasks/           — Task definitions (L1–L4) and registry
agents/          — LLM client and tool dispatcher
workflows/       — The 10 workflow pattern implementations
evaluation/      — Metrics aggregation and LLM-as-judge scoring
scripts/         — CLI (Typer)
data/csv/        — Source CSV dataset
tests/           — Pytest test suite (153 tests)
results/         — Benchmark output (gitignored)
plots/           — Generated charts (gitignored)
```

## Architecture

All workflows share the same base classes:

- `workflows/base.py` — `BaseWorkflow` and `WorkflowResult`
- `tools/base.py` — `BaseTool` and `ToolResult`
- `agents/llm_client.py` — `LLMClient` (provider-agnostic, retry/timeout handling)
- `tasks/task_registry.py` — `Task`, `TaskLevel`, `TaskRegistry`
- `evaluation/metrics.py` — `WorkflowMetrics`

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Surabhi Pradhan.
