"""Conditional routing logic."""

# src/graph/edges.py

from src.state.schema import CoachState


def route_from_supervisor(state: CoachState) -> str:
    return state.get("next_agent", "responder")


def route_from_evaluator(state: CoachState) -> str:
    evaluation = state.get("evaluation")
    if evaluation is None:
        return "continue"
    if evaluation.escalate_to_human:
        return "hitl"
    if not evaluation.passed and state["error_count"] < 3:
        return "retry"
    return "continue"


def route_hitl_check(state: CoachState) -> str:
    return "hitl" if state.get("hitl_required") else "memory_write"


def route_finetune_check(state: CoachState) -> str:
    return "trigger_finetune" if state.get("finetune_trigger") else "skip"