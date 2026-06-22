import logging
import os
import sys
from contextvars import ContextVar
from typing import Any, Dict

import structlog
from structlog.types import EventDict, Processor

# Context variables for async/thread-safe tracing
user_id_var: ContextVar[str] = ContextVar("user_id", default="unknown")
session_id_var: ContextVar[str] = ContextVar("session_id", default="unknown")
node_name_var: ContextVar[str] = ContextVar("node_name", default="unknown")
crew_name_var: ContextVar[str] = ContextVar("crew_name", default="unknown")

def add_custom_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add contextvars to the event dict."""
    event_dict["user_id"] = user_id_var.get()
    event_dict["session_id"] = session_id_var.get()
    event_dict["node_name"] = node_name_var.get()
    event_dict["crew_name"] = crew_name_var.get()
    return event_dict

def configure_structlog():
    """
    Configure structlog with production-ready defaults:
    - Async-safe context via contextvars.
    - JSON output for production, Pretty Console for development.
    - High-quality timestamps and level tagging.
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_custom_context,
        structlog.processors.UnicodeDecoder(),
    ]

    # Environment-based renderer
    if os.getenv("APP_ENV", "development").lower() == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory() if os.getenv("APP_ENV") == "production" else structlog.WriteLoggerFactory(sys.stderr),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
        ),
        cache_logger_on_first_use=True,
    )

def bind_context(**kwargs: Any) -> None:
    """Bind multiple context variables at once."""
    if "user_id" in kwargs:
        user_id_var.set(str(kwargs["user_id"]))
    if "session_id" in kwargs:
        session_id_var.set(str(kwargs["session_id"]))
    if "node_name" in kwargs:
        node_name_var.set(str(kwargs["node_name"]))
    if "crew_name" in kwargs:
        crew_name_var.set(str(kwargs["crew_name"]))

def clear_context() -> None:
    """Clear all bound context variables."""
    user_id_var.set("unknown")
    session_id_var.set("unknown")
    node_name_var.set("unknown")
    crew_name_var.set("unknown")