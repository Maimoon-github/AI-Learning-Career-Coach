from __future__ import annotations

from typing import Any

import structlog
from langgraph.types import interrupt

from src.langgraph_workflow.state import AgentState

log = structlog.get_logger(__name__)


async def hitl_node(state: AgentState) -> dict[str, Any]:
    log.info("node.hitl.waiting", user_id=state["user_id"], week=state.get("current_week", 1))

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