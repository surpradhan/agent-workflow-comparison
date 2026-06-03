"""LLM client wrapper supporting Anthropic and OpenAI via LangChain."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import BaseMessage

from config import settings

log = logging.getLogger(__name__)

# Retry / timeout configuration
_RETRY_DELAYS = (2.0, 4.0, 8.0)   # seconds between successive retries
_LLM_TIMEOUT = 180.0               # seconds before a single call is abandoned (generous for local Ollama; Groq is much faster)

# Typed exception classes from provider SDKs — preferred over keyword matching.
# Guarded by try/except so the module loads even when a provider is not installed.
try:
    from anthropic import APIConnectionError as _AnthrConnErr
    from anthropic import APIStatusError as _AnthrStatusErr
    from anthropic import RateLimitError as _AnthrRateLimitErr
    # RateLimitErr and ConnectionErr are retriable; APIStatusError covers all HTTP
    # responses — some (429, 503) are retriable, others (401, 403) are permanent.
    # We don't blanket-mark APIStatusError as retriable; the keyword fallback handles
    # the retriable HTTP codes while the permanent-keyword guard rejects auth errors.
    _ANTHROPIC_RETRIABLE: tuple = (_AnthrRateLimitErr, _AnthrConnErr)
    _ANTHROPIC_PERMANENT: tuple = (_AnthrStatusErr,)
except ImportError:
    _ANTHROPIC_RETRIABLE = ()
    _ANTHROPIC_PERMANENT = ()

try:
    from openai import APIConnectionError as _OAIConnErr
    from openai import RateLimitError as _OAIRateLimitErr
    _OPENAI_RETRIABLE: tuple = (_OAIRateLimitErr, _OAIConnErr)
except ImportError:
    _OPENAI_RETRIABLE = ()

_TYPED_RETRIABLE: tuple = _ANTHROPIC_RETRIABLE + _OPENAI_RETRIABLE

# Keyword fallback for un-typed / wrapped exceptions.
# "connection" removed — too broad, matches auth/SSL errors.
_RETRIABLE_KEYWORDS: tuple[str, ...] = (
    "rate limit", "overload", "timeout",
    "503", "529", "too many", "service unavailable",
)

# Any exception whose message contains these strings is a permanent failure —
# never retry regardless of other keyword matches.
_PERMANENT_KEYWORDS: tuple[str, ...] = (
    "authentication", "api key", "unauthorized", "forbidden",
    "invalid api", "permission denied",
)


def _is_retriable(exc: Exception) -> bool:
    """Return True if this exception is worth retrying."""
    msg = str(exc).lower()

    # Explicit permanent-failure guard wins over everything else.
    # Catches auth errors (401/403) regardless of exception type.
    if any(kw in msg for kw in _PERMANENT_KEYWORDS):
        return False

    # Typed check: known permanent HTTP-error base classes (e.g. APIStatusError)
    # are not retriable unless they are also a known retriable subclass.
    if _ANTHROPIC_PERMANENT and isinstance(exc, _ANTHROPIC_PERMANENT):
        if not (_ANTHROPIC_RETRIABLE and isinstance(exc, _ANTHROPIC_RETRIABLE)):
            # It's a permanent HTTP error (e.g. 400 Bad Request) — don't retry.
            # Exception: still retry if the message contains a retriable keyword
            # (e.g. APIStatusError wrapping a 503 response).
            return any(kw in msg for kw in _RETRIABLE_KEYWORDS)

    # Typed retriable classes (RateLimitError, APIConnectionError)
    if _TYPED_RETRIABLE and isinstance(exc, _TYPED_RETRIABLE):
        return True

    # Keyword fallback for un-typed / wrapped exceptions
    return any(kw in msg for kw in _RETRIABLE_KEYWORDS)


def extract_text(response: Any) -> str:
    """Extract plain text from a LangChain chat message.

    Exposed as a module-level function so other modules (e.g. tool_using)
    can import it instead of duplicating the logic.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Anthropic sometimes returns a list of content blocks
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "\n".join(p for p in parts if p)
    return str(content)


def _build_model(model_override: str | None = None) -> Any:
    """Build the appropriate LangChain chat model from settings.

    Args:
        model_override: Use this model name instead of settings.llm_model.
                        Used by the judge when settings.judge_model is set.
    """
    model_name = model_override or settings.llm_model

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
        )
    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
        )
    elif settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            api_key=settings.groq_api_key,
        )
    else:
        raise ValueError(
            f"Unknown llm_provider '{settings.llm_provider}'. "
            "Must be 'anthropic', 'openai', 'ollama', or 'groq'."
        )


class LLMClient:
    """Thin async wrapper around a LangChain chat model.

    Features:
    - Automatic retry with exponential backoff for transient errors
    - Per-call timeout to prevent indefinite hangs
    - Unified token extraction from usage_metadata
    - Plain-text and tool-bound invocation modes
    """

    def __init__(self, model_override: str | None = None) -> None:
        """
        Args:
            model_override: Use a specific model name instead of settings.llm_model.
                            Passed through to _build_model; useful for the judge
                            when settings.judge_model differs from llm_model.
        """
        self._model = _build_model(model_override=model_override)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tokens(response: Any) -> int:
        """Pull total_tokens from usage_metadata, safely handling None values."""
        meta = getattr(response, "usage_metadata", None)
        if meta:
            return int(meta.get("total_tokens") or 0)  # guard against explicit None
        return 0

    async def _call_with_retry(self, coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke an async callable with per-call timeout and exponential backoff.

        Retries up to len(_RETRY_DELAYS) times on transient errors.
        Non-retriable exceptions are re-raised immediately.
        Sets self._last_retry_count to the number of retries used (0 = first attempt succeeded).
        """
        self._last_retry_count = 0
        last_exc: Exception | None = None
        for attempt, delay in enumerate([0.0, *_RETRY_DELAYS]):
            if delay:
                log.warning(
                    "LLM call attempt %d/%d — waiting %.1fs before retry",
                    attempt + 1, len(_RETRY_DELAYS) + 1, delay,
                )
                await asyncio.sleep(delay)
            try:
                result = await asyncio.wait_for(
                    coro_fn(*args, **kwargs), timeout=_LLM_TIMEOUT
                )
                self._last_retry_count = attempt  # 0 = no retry needed
                return result
            except TimeoutError as exc:
                log.warning(
                    "LLM call timed out after %.0fs (attempt %d)", _LLM_TIMEOUT, attempt + 1
                )
                last_exc = exc
            except Exception as exc:
                if _is_retriable(exc):
                    log.warning("Retriable LLM error on attempt %d: %s", attempt + 1, exc)
                    last_exc = exc
                else:
                    raise  # non-transient: propagate immediately
        assert last_exc is not None
        raise last_exc

    @property
    def last_retries(self) -> int:
        """Number of retries used on the most recent call (0 = succeeded first try)."""
        return getattr(self, "_last_retry_count", 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def invoke(self, messages: list[BaseMessage]) -> tuple[str, int]:
        """Send messages and return (text_response, total_tokens)."""
        response = await self._call_with_retry(self._model.ainvoke, messages)
        return extract_text(response), self._extract_tokens(response)

    async def invoke_with_tools(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]],
    ) -> tuple[Any, int]:
        """Send messages with tool schemas bound.

        Returns (AIMessage, total_tokens). Caller inspects
        ``response.tool_calls`` to detect tool-use requests.
        """
        model = self._model.bind_tools(tools)
        response = await self._call_with_retry(model.ainvoke, messages)
        return response, self._extract_tokens(response)
