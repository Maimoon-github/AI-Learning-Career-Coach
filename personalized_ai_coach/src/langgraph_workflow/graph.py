from __future__ import annotations

import os
from typing import Literal

import structlog
import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.langgraph_workflow.nodes.profile_ingestion_node import profile_ingestion_node
from src.langgraph_workflow.nodes.skill_assessment_node import skill_assessment_node
from src.langgraph_workflow.nodes.learning_path_node import learning_path_node
from src.langgraph_workflow.nodes.project_generation_node import project_generation_node
from src.langgraph_workflow.nodes.llm_fine_tuning_node import llm_fine_tuning_node
from src.langgraph_workflow.nodes.progress_report_node import progress_report_node
from src.langgraph_workflow.nodes.hitl_node import hitl_node
from src.langgraph_workflow.state import AgentState

log = structlog.get_logger(__name__)


def _load_settings() -> dict:
    with open("config/system_settings.yaml") as f:
        return yaml.safe_load(f)


def route_after_profile(state: AgentState) -> Literal["skill_assessment", END]:
    if state.get("error_context") and not state.get("skill_profile"):
        log.error("routing.profile_failed", error=state["error_context"])
        return END
    return "skill_assessment"


def route_after_hitl(state: AgentState) -> Literal["learning_path", END]:
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
        return "learning_path"
    # approve → advance week
    log.info("routing.hitl.approved", week=state.get("current_week"))
    return "learning_path"


def route_after_report(state: AgentState) -> Literal["hitl", END]:
    return "hitl"  # always go to HITL for review


def _advance_week(state: AgentState) -> dict:
    return {"current_week": state.get("current_week", 1) + 1, "hitl_action": None}


def build_graph(checkpointer=None) -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("profile_ingestion", profile_ingestion_node)
    workflow.add_node("skill_assessment", skill_assessment_node)
    workflow.add_node("learning_path", learning_path_node)
    workflow.add_node("project_generation", project_generation_node)
    workflow.add_node("llm_fine_tuning", llm_fine_tuning_node)
    workflow.add_node("progress_report", progress_report_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("advance_week", _advance_week)

    # Entry
    workflow.add_edge(START, "profile_ingestion")

    # Conditional: profile → skill_assessment or END
    workflow.add_conditional_edges(
        "profile_ingestion",
        route_after_profile,
        {"skill_assessment": "skill_assessment", END: END},
    )

    # Linear
    workflow.add_edge("skill_assessment", "learning_path")

    # Parallel fan-out from learning_path
    workflow.add_edge("learning_path", "project_generation")
    workflow.add_edge("learning_path", "llm_fine_tuning")

    # Both parallel branches converge to progress_report
    workflow.add_edge("project_generation", "progress_report")
    workflow.add_edge("llm_fine_tuning", "progress_report")

    # Report → HITL
    workflow.add_conditional_edges(
        "progress_report",
        route_after_report,
        {"hitl": "hitl", END: END},
    )

    # HITL → advance_week then back to learning_path, or END
    workflow.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {"learning_path": "advance_week", END: END},
    )
    workflow.add_edge("advance_week", "learning_path")

    return workflow


def _create_redis_checkpointer():
    try:
        from langgraph.checkpoint.redis import RedisSaver
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return RedisSaver.from_conn_string(redis_url)
    except ImportError:
        log.warning("redis_checkpointer_unavailable, using memory")
        return MemorySaver()


def _create_postgres_checkpointer():
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        db_url = os.getenv("DATABASE_URL", "").replace("+asyncpg", "")
        return AsyncPostgresSaver.from_conn_string(db_url)
    except ImportError:
        log.warning("postgres_checkpointer_unavailable, using memory")
        return MemorySaver()


def create_app(backend: str = "memory"):
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