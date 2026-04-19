"""Shared utilities for workflow implementations."""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# Single constant used by all workflows — keeps tool result context manageable
# without silently discarding too much data.
MAX_TOOL_RESULT_CHARS = 3000


def parse_json(text: str) -> tuple[dict, bool]:
    """Parse a JSON object/array from an LLM response, tolerating common noise.

    Returns (data, parse_ok).  parse_ok=False means all strategies failed;
    data will be an empty dict in that case.

    Strategies tried in order:
    1. Direct parse (clean JSON).
    2. Strip markdown fences (``` or ```json ... ```).
    3. Extract the first {...} or [...] block from within prose — handles
       "Here is my plan: {...}" and trailing explanations after the closing brace.
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip()), True
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract first {...} or [...] block
    # Scan for the opening brace/bracket, then walk forward tracking depth.
    for opener, closer in (('{', '}'), ('[', ']')):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate), True
                    except json.JSONDecodeError:
                        break  # malformed — try next opener type

    return {}, False
