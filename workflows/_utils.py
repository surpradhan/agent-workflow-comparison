"""Shared utilities for workflow implementations."""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

# Single constant used by all workflows — keeps tool result context manageable
# without silently discarding too much data.
MAX_TOOL_RESULT_CHARS = 3000


def parse_json(text: str) -> tuple[dict, bool]:
    """Parse a JSON string returned by an LLM, tolerating markdown code fences.

    Returns (data, parse_ok).  parse_ok=False means JSON decoding failed;
    data will be an empty dict in that case.

    Handles triple-backtick fences with or without a language tag, e.g.:
        ```json
        { ... }
        ```
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line (may be "```" or "```json")
        # Drop the closing fence line if it is exactly "```"
        inner_start = 1
        inner_end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[inner_start:inner_end])
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return {}, False
