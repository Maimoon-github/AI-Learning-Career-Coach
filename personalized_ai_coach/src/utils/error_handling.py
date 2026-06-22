import asyncio
import functools
import random
import time
from typing import Any, Callable, ParamSpec, TypeVar

import structlog

log = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# --- Domain-Specific Exceptions ---

class CoachError(Exception):
    """Base exception for all Coach utilities."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}

class ToolExecutionError(CoachError):
    """Raised when an external tool (GitHub, Kaggle, Search) fails."""
    pass

class CrewExecutionError(CoachError):
    """Raised when a multi-agent crew fails to complete a task."""
    pass

class LLMProviderError(CoachError):
    """Raised when the LLM provider (Ollama) is unreachable or errors."""
    pass

class StructuredExtractionError(CoachError):
    """Raised when structured output parsing fails."""
    pass

class ValidationError(CoachError):
    """Raised when business logic or schema validation fails."""
    pass

# --- Retry Logic with Jitter ---

def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Standard retry decorator with exponential backoff and optional jitter.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        log.error("retry_final_failure", func=func.__name__, attempt=attempt, error=str(e))
                        raise
                    
                    wait_time = delay
                    if jitter:
                        wait_time *= (0.5 + random.random())
                    
                    log.warning(
                        "retry_attempt_failed",
                        func=func.__name__,
                        attempt=attempt,
                        next_retry_in=f"{wait_time:.2f}s",
                        error=str(e)
                    )
                    time.sleep(wait_time)
                    delay *= backoff_factor
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

def async_retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Async retry decorator with exponential backoff and optional jitter.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        log.error("async_retry_final_failure", func=func.__name__, attempt=attempt, error=str(e))
                        raise
                    
                    wait_time = delay
                    if jitter:
                        wait_time *= (0.5 + random.random())
                    
                    log.warning(
                        "async_retry_attempt_failed",
                        func=func.__name__,
                        attempt=attempt,
                        next_retry_in=f"{wait_time:.2f}s",
                        error=str(e)
                    )
                    await asyncio.sleep(wait_time)
                    delay *= backoff_factor
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator