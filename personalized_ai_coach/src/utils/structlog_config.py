import structlog
import os
from contextvars import ContextVar

user_id_var = ContextVar("user_id", default="unknown")
session_id_var = ContextVar("session_id", default="unknown")
node_name_var = ContextVar("node_name", default="unknown")
crew_name_var = ContextVar("crew_name", default="unknown")

def configure_structlog():
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    if os.getenv("APP_ENV") == "development":
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]
    else:
        processors = shared_processors + [structlog.processors.JSONRenderer()]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), os.getenv("LOG_LEVEL", "INFO"))
        ),
    )

def bind_context(user_id=None, session_id=None, node_name=None, crew_name=None):
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)
    if node_name:
        node_name_var.set(node_name)
    if crew_name:
        crew_name_var.set(crew_name)