import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

# Custom exceptions
class ToolExecutionError(Exception):
    """Raised when a tool (GitHub, Kaggle, web search) fails."""
    pass

class CrewExecutionError(Exception):
    """Raised when a CrewAI crew fails after retries."""
    pass

class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable or returns an error."""
    pass

class ValidationError(Exception):
    """Raised when Pydantic validation fails on agent output."""
    pass

class HITLTimeoutError(Exception):
    """Raised when a human-in-the-loop gate times out."""
    pass

P = ParamSpec("P")
T = TypeVar("T")

def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that retries a function with exponential backoff."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"Final attempt {attempt} failed: {e}")
                        raise
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay}s")
                    time.sleep(delay)
                    delay *= backoff_multiplier
            raise RuntimeError("Unreachable")  # pragma: no cover
        return wrapper
    return decorator

# Async version
def async_retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"Final async attempt {attempt} failed: {e}")
                        raise
                    logger.warning(f"Async attempt {attempt} failed: {e}. Retrying in {delay}s")
                    await asyncio.sleep(delay)
                    delay *= backoff_multiplier
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator