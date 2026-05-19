from __future__ import annotations

import functools
import asyncio
from typing import Any, Callable, TypeVar

import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")


class CoachBaseError(Exception):
    """Base error for all application errors."""
    code: str = "UNKNOWN_ERROR"

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class ToolExecutionError(CoachBaseError):
    code = "TOOL_EXECUTION_ERROR"


class ParsingError(CoachBaseError):
    code = "PARSING_ERROR"


class ContextLimitExceeded(CoachBaseError):
    code = "CONTEXT_LIMIT_EXCEEDED"


class CrewExecutionError(CoachBaseError):
    code = "CREW_EXECUTION_ERROR"


class ValidationError(CoachBaseError):
    code = "VALIDATION_ERROR"


class HITLTimeoutError(CoachBaseError):
    code = "HITL_TIMEOUT"


class OllamaConnectionError(CoachBaseError):
    code = "OLLAMA_CONNECTION_ERROR"


class FineTuningError(CoachBaseError):
    code = "FINE_TUNING_ERROR"


ERROR_TAXONOMY: dict[str, type[CoachBaseError]] = {
    cls.code: cls  # type: ignore[attr-defined]
    for cls in [
        ToolExecutionError, ParsingError, ContextLimitExceeded,
        CrewExecutionError, ValidationError, HITLTimeoutError,
        OllamaConnectionError, FineTuningError,
    ]
}


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    retriable_errors: tuple[type[Exception], ...] = (ToolExecutionError, CrewExecutionError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Async retry decorator with exponential backoff."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except retriable_errors as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        delay = backoff_base ** (attempt - 1)
                        log.warning(
                            "retrying_after_error",
                            fn=fn.__name__,
                            attempt=attempt,
                            delay=delay,
                            error=str(exc),
                        )
                        await asyncio.sleep(delay)
                except Exception:
                    raise
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator