from __future__ import annotations

import time
from typing import Any

import structlog
from langgraph.types import interrupt

from src.langgraph_workflow.state import AgentState
from src.utils.error_handling import HITLTimeoutError

log = structlog.get_logger(__name__)

# Default HITL deadline timeout in seconds (0 = disabled)
_HITL_TIMEOUT_SECONDS = 0


def _check_hitl_timeout(state: AgentState) -> None:
    """Raise HITLTimeoutError if a stored deadline has passed.

    The deadline is stored in state as ``hitl_deadline_ts`` (Unix timestamp,
    float).  If the key is absent or set to None the check is skipped, which
    preserves backward-compatibility for callers that do not set a deadline.
    """
    deadline = state.get("hitl_deadline_ts")  # type: ignore[attr-defined]
    if deadline is not None and time.time() > deadline:
        raise HITLTimeoutError(
            f"HITL gate timed out after deadline={deadline:.1f} "
            f"(now={time.time():.1f})"
        )


async def hitl_node(state: AgentState) -> dict[str, Any]:
    log.info("node.hitl.waiting", user_id=state["user_id"], week=state.get("current_week", 1))

    # Enforce deadline if one was set upstream
    _check_hitl_timeout(state)

    presentation = {
        "weekly_report": state.get("weekly_report"),
        "learning_path_preview": (state.get("learning_path") or {}).get("weeks", [])[:2],
        "projects_preview": state.get("practice_projects", [])[:3],
        "current_week": state.get("current_week", 1),
        "revision_cycle": state.get("revision_cycle", 0),
    }

    # Suspend workflow until user provides input
    user_response = interrupt(value=presentation)

    # Expected payload: {"hitl_action": "approve"|"revise"|"end", "user_feedback": str|None}
    if isinstance(user_response, dict):
        action = user_response.get("hitl_action", "approve")
        feedback = user_response.get("user_feedback")
    else:
        action = "approve"
        feedback = None

    log.info("node.hitl.resumed", user_id=state["user_id"], action=action)

    return {
        "hitl_action": action,
        "user_feedback": feedback,
        "revision_cycle": state.get("revision_cycle", 0) + (1 if action == "revise" else 0),
    }