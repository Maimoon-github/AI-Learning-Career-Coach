from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Identity
    user_id: str
    target_role: str
    session_id: str

    # Inputs
    github_profile_url: str | None
    kaggle_username: str | None
    uploaded_document_paths: list[str]
    session_notes: list[str]

    # Crew outputs
    skill_profile: dict[str, Any] | None
    skill_gaps: list[dict[str, Any]]
    learning_path: dict[str, Any] | None
    practice_projects: list[dict[str, Any]]
    fine_tuning_status: str | None
    fine_tuning_metrics: dict[str, Any] | None
    weekly_report: dict[str, Any] | None

    # Workflow control
    current_week: int
    revision_cycle: int
    user_feedback: str | None
    hitl_action: str | None
    error_context: dict[str, Any] | None

    # Conversation (LangGraph reducer)
    messages: Annotated[list[BaseMessage], add_messages]


def initial_state(
    user_id: str,
    target_role: str,
    session_id: str,
    github_profile_url: str | None = None,
    kaggle_username: str | None = None,
    uploaded_document_paths: list[str] | None = None,
) -> AgentState:
    return AgentState(
        user_id=user_id,
        target_role=target_role,
        session_id=session_id,
        github_profile_url=github_profile_url,
        kaggle_username=kaggle_username,
        uploaded_document_paths=uploaded_document_paths or [],
        session_notes=[],
        skill_profile=None,
        skill_gaps=[],
        learning_path=None,
        practice_projects=[],
        fine_tuning_status=None,
        fine_tuning_metrics=None,
        weekly_report=None,
        current_week=1,
        revision_cycle=0,
        user_feedback=None,
        hitl_action=None,
        error_context=None,
        messages=[],
    )