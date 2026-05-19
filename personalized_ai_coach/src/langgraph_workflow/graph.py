from __future__ import annotations

import os
from typing import Literal

import structlog
import yaml
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.langgraph_workflow.nodes.all_nodes import (
    hitl_node,
    learning_path_node,
    llm_fine_tuning_node,
    profile_ingestion_node,
    progress_report_node,
    project_generation_node,
    skill_assessment_node,
)
from src.langgraph_workflow.state import AgentState

log = structlog.get_logger(__name__)


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_profile(state: AgentState) -> Literal["skill_assessment", END]:
    if state.get("error_context") and not state.get("skill_profile"):
        log.error("routing.profile_failed", error=state["error_context"])
        return END
    return "skill_assessment"


def route_after_hitl(
    state: AgentState,
) -> Literal["learning_path", END]:
    settings = _load_settings()
    action = state.get("hitl_action", "approve")
    revision_cycle = state.get("revision_cycle", 0)
    max_revisions = settings["hitl"]["max_revision_cycles"]

    if action == "end":
        log.info("routing.hitl.end_session")
        return END

    if action == "revise" and revision_cycle <= max_revisions:
        log.info("routing.hitl.revise", cycle=revision_cycle)
        return "learning_path"

    if action == "revise" and revision_cycle > max_revisions:
        log.warning("routing.hitl.max_revisions_exceeded", forcing_approve=True)
        return "learning_path"  # One final attempt

    # approve → advance week
    log.info("routing.hitl.approved", week=state.get("current_week"))
    return "learning_path"  # Loop continues for next week


def route_after_report(state: AgentState) -> Literal["hitl", END]:
    """Always go to HITL after report — user must review each week."""
    current_week = state.get("current_week", 1)
    total_weeks = len((state.get("learning_path") or {}).get("weeks", []))
    if current_week >= total_weeks and total_weeks > 0:
        log.info("routing.report.final_week_complete", week=current_week)
    return "hitl"


def _load_settings() -> dict:
    with open("config/system_settings.yaml") as f:
        return yaml.safe_load(f)


def _advance_week(state: AgentState) -> dict:
    """Increment week counter after HITL approval."""
    return {"current_week": state.get("current_week", 1) + 1, "hitl_action": None}


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the full LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("profile_ingestion", profile_ingestion_node)
    workflow.add_node("skill_assessment", skill_assessment_node)
    workflow.add_node("learning_path", learning_path_node)
    workflow.add_node("project_generation", project_generation_node)
    workflow.add_node("llm_fine_tuning", llm_fine_tuning_node)
    workflow.add_node("progress_report", progress_report_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("advance_week", _advance_week)

    # ── Entry ─────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "profile_ingestion")

    # ── Conditional: profile → skill_assessment or END on failure ─────────────
    workflow.add_conditional_edges(
        "profile_ingestion",
        route_after_profile,
        {"skill_assessment": "skill_assessment", END: END},
    )

    # ── Linear: skill_assessment → learning_path ──────────────────────────────
    workflow.add_edge("skill_assessment", "learning_path")

    # ── Parallel fan-out from learning_path ───────────────────────────────────
    workflow.add_edge("learning_path", "project_generation")
    workflow.add_edge("learning_path", "llm_fine_tuning")

    # ── Both parallel branches → progress_report ─────────────────────────────
    # LangGraph will wait for both to complete before proceeding
    workflow.add_edge("project_generation", "progress_report")
    workflow.add_edge("llm_fine_tuning", "progress_report")

    # ── progress_report → hitl (always) ──────────────────────────────────────
    workflow.add_conditional_edges(
        "progress_report",
        route_after_report,
        {"hitl": "hitl", END: END},
    )

    # ── HITL → advance_week then back to learning_path OR end ─────────────────
    workflow.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {"learning_path": "advance_week", END: END},
    )
    workflow.add_edge("advance_week", "learning_path")

    return workflow


def create_app(backend: str = "memory"):
    """Compile the graph with the appropriate checkpointing backend."""
    if backend == "redis":
        checkpointer = _create_redis_checkpointer()
    elif backend == "postgres":
        checkpointer = _create_postgres_checkpointer()
    else:
        checkpointer = MemorySaver()

    graph = build_graph(checkpointer=checkpointer)
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["hitl"])
    log.info("graph.compiled", backend=backend)
    return app


def _create_redis_checkpointer():
    """Create a Redis-backed checkpointer for production persistence."""
    try:
        from langgraph.checkpoint.redis import RedisSaver
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return RedisSaver.from_conn_string(redis_url)
    except ImportError:
        log.warning("redis_checkpointer_unavailable", fallback="memory")
        return MemorySaver()


def _create_postgres_checkpointer():
    """Create a PostgreSQL-backed checkpointer for production persistence."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        db_url = os.getenv("DATABASE_URL", "").replace("+asyncpg", "")
        return AsyncPostgresSaver.from_conn_string(db_url)
    except ImportError:
        log.warning("postgres_checkpointer_unavailable", fallback="memory")
        return MemorySaver()





























from __future__ import annotations

import os
from typing import Literal

import structlog
import yaml
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.langgraph_workflow.nodes.all_nodes import (
    hitl_node,
    learning_path_node,
    llm_fine_tuning_node,
    profile_ingestion_node,
    progress_report_node,
    project_generation_node,
    skill_assessment_node,
)
from src.langgraph_workflow.state import AgentState

log = structlog.get_logger(__name__)


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_profile(state: AgentState) -> Literal["skill_assessment", END]:
    if state.get("error_context") and not state.get("skill_profile"):
        log.error("routing.profile_failed", error=state["error_context"])
        return END
    return "skill_assessment"


def route_after_hitl(
    state: AgentState,
) -> Literal["learning_path", END]:
    settings = _load_settings()
    action = state.get("hitl_action", "approve")
    revision_cycle = state.get("revision_cycle", 0)
    max_revisions = settings["hitl"]["max_revision_cycles"]

    if action == "end":
        log.info("routing.hitl.end_session")
        return END

    if action == "revise" and revision_cycle <= max_revisions:
        log.info("routing.hitl.revise", cycle=revision_cycle)
        return "learning_path"

    if action == "revise" and revision_cycle > max_revisions:
        log.warning("routing.hitl.max_revisions_exceeded", forcing_approve=True)
        return "learning_path"  # One final attempt

    # approve → advance week
    log.info("routing.hitl.approved", week=state.get("current_week"))
    return "learning_path"  # Loop continues for next week


def route_after_report(state: AgentState) -> Literal["hitl", END]:
    """Always go to HITL after report — user must review each week."""
    current_week = state.get("current_week", 1)
    total_weeks = len((state.get("learning_path") or {}).get("weeks", []))
    if current_week >= total_weeks and total_weeks > 0:
        log.info("routing.report.final_week_complete", week=current_week)
    return "hitl"


def _load_settings() -> dict:
    with open("config/system_settings.yaml") as f:
        return yaml.safe_load(f)


def _advance_week(state: AgentState) -> dict:
    """Increment week counter after HITL approval."""
    return {"current_week": state.get("current_week", 1) + 1, "hitl_action": None}


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the full LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("profile_ingestion", profile_ingestion_node)
    workflow.add_node("skill_assessment", skill_assessment_node)
    workflow.add_node("learning_path", learning_path_node)
    workflow.add_node("project_generation", project_generation_node)
    workflow.add_node("llm_fine_tuning", llm_fine_tuning_node)
    workflow.add_node("progress_report", progress_report_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("advance_week", _advance_week)

    # ── Entry ─────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "profile_ingestion")

    # ── Conditional: profile → skill_assessment or END on failure ─────────────
    workflow.add_conditional_edges(
        "profile_ingestion",
        route_after_profile,
        {"skill_assessment": "skill_assessment", END: END},
    )

    # ── Linear: skill_assessment → learning_path ──────────────────────────────
    workflow.add_edge("skill_assessment", "learning_path")

    # ── Parallel fan-out from learning_path ───────────────────────────────────
    workflow.add_edge("learning_path", "project_generation")
    workflow.add_edge("learning_path", "llm_fine_tuning")

    # ── Both parallel branches → progress_report ─────────────────────────────
    # LangGraph will wait for both to complete before proceeding
    workflow.add_edge("project_generation", "progress_report")
    workflow.add_edge("llm_fine_tuning", "progress_report")

    # ── progress_report → hitl (always) ──────────────────────────────────────
    workflow.add_conditional_edges(
        "progress_report",
        route_after_report,
        {"hitl": "hitl", END: END},
    )

    # ── HITL → advance_week then back to learning_path OR end ─────────────────
    workflow.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {"learning_path": "advance_week", END: END},
    )
    workflow.add_edge("advance_week", "learning_path")

    return workflow


def create_app(backend: str = "memory"):
    """Compile the graph with the appropriate checkpointing backend."""
    if backend == "redis":
        checkpointer = _create_redis_checkpointer()
    elif backend == "postgres":
        checkpointer = _create_postgres_checkpointer()
    else:
        checkpointer = MemorySaver()

    graph = build_graph(checkpointer=checkpointer)
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["hitl"])
    log.info("graph.compiled", backend=backend)
    return app


def _create_redis_checkpointer():
    """Create a Redis-backed checkpointer for production persistence."""
    try:
        from langgraph.checkpoint.redis import RedisSaver
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return RedisSaver.from_conn_string(redis_url)
    except ImportError:
        log.warning("redis_checkpointer_unavailable", fallback="memory")
        return MemorySaver()


def _create_postgres_checkpointer():
    """Create a PostgreSQL-backed checkpointer for production persistence."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        db_url = os.getenv("DATABASE_URL", "").replace("+asyncpg", "")
        return AsyncPostgresSaver.from_conn_string(db_url)
    except ImportError:
        log.warning("postgres_checkpointer_unavailable", fallback="memory")
        return MemorySaver()