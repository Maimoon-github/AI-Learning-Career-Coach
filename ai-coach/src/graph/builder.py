"""StateGraph construction."""

# src/graph/builder.py

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state.schema import CoachState
from src.graph.nodes import (
    memory_ingest_node,
    rag_node,
    voice_output_node,
    memory_write_node,
    finetune_check_node,
    responder_node,
)
from src.agents.supervisor import supervisor_node
from src.agents.profile_analyst import run_profile_analyst
from src.agents.curriculum_planner import run_curriculum_planner
from src.agents.project_builder import run_project_builder
from src.agents.evaluator import evaluator_node
from src.agents.reporter import run_reporter
from src.graph.edges import (
    route_from_supervisor,
    route_from_evaluator,
    route_hitl_check,
    route_finetune_check,
)
from src.graph.checkpointer import get_checkpointer


def build_coach_graph() -> StateGraph:
    """Construct and compile the full LangGraph StateGraph."""

    graph = StateGraph(CoachState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("memory_ingest",    memory_ingest_node)
    graph.add_node("rag",              rag_node)
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("profile_analyst",  run_profile_analyst)
    graph.add_node("curriculum_planner", run_curriculum_planner)
    graph.add_node("project_builder",  run_project_builder)
    graph.add_node("evaluator",        evaluator_node)
    graph.add_node("reporter",         run_reporter)
    graph.add_node("responder",        responder_node)
    graph.add_node("hitl",             hitl_node)           # interrupt point
    graph.add_node("memory_write",     memory_write_node)
    graph.add_node("finetune_check",   finetune_check_node)
    graph.add_node("voice_output",     voice_output_node)

    # ── Edges: start ──────────────────────────────────────────────
    graph.add_edge(START, "memory_ingest")
    graph.add_edge("memory_ingest", "rag")
    graph.add_edge("rag", "supervisor")

    # ── Edges: supervisor routes ──────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "profile_analyst":   "profile_analyst",
            "curriculum_planner":"curriculum_planner",
            "project_builder":   "project_builder",
            "reporter":          "reporter",
            "responder":         "responder",
        }
    )

    # ── Edges: agents → evaluator ─────────────────────────────────
    graph.add_edge("profile_analyst",   "evaluator")
    graph.add_edge("curriculum_planner","evaluator")
    graph.add_edge("project_builder",   "evaluator")

    # ── Edges: evaluator conditional ─────────────────────────────
    graph.add_conditional_edges(
        "evaluator",
        route_from_evaluator,
        {
            "retry":     "supervisor",    # score < 0.7, retries remain
            "hitl":      "hitl",          # escalate_to_human
            "continue":  "hitl_check",    # passed
        }
    )

    # ── Edges: HITL check ─────────────────────────────────────────
    graph.add_node("hitl_check", lambda s: s)   # passthrough decision node
    graph.add_conditional_edges(
        "hitl_check",
        route_hitl_check,
        {
            "hitl":         "hitl",
            "memory_write": "memory_write",
        }
    )

    # ── Edges: HITL → resume ──────────────────────────────────────
    graph.add_conditional_edges(
        "hitl",
        lambda s: "memory_write" if s.get("human_approval") else "supervisor",
        {
            "memory_write": "memory_write",
            "supervisor":   "supervisor",
        }
    )

    # ── Edges: report + responder bypass evaluator ────────────────
    graph.add_edge("reporter",  "memory_write")
    graph.add_edge("responder", "memory_write")

    # ── Edges: memory → finetune check → voice → END ─────────────
    graph.add_edge("memory_write", "finetune_check")
    graph.add_conditional_edges(
        "finetune_check",
        route_finetune_check,
        {
            "trigger_finetune": "voice_output",   # background job launched
            "skip":             "voice_output",
        }
    )
    graph.add_edge("voice_output", END)

    # ── Compile with checkpointer ─────────────────────────────────
    checkpointer = get_checkpointer()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],   # LangGraph native HITL breakpoint
    )