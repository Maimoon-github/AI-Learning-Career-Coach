from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    user_id: str
    target_role: str
    session_id: str

    # ── Inputs ────────────────────────────────────────────────────────────────
    github_profile_url: str | None
    kaggle_username: str | None
    uploaded_document_paths: list[str]
    session_notes: list[str]

    # ── Crew outputs ─────────────────────────────────────────────────────────
    skill_profile: dict[str, Any] | None          # SkillProfile.model_dump()
    skill_gaps: list[dict[str, Any]]               # list[SkillGap.model_dump()]
    learning_path: dict[str, Any] | None           # LearningPath.model_dump()
    practice_projects: list[dict[str, Any]]        # list[ProjectSpec.model_dump()]
    fine_tuning_status: str | None                 # pending|running|complete|failed
    fine_tuning_metrics: dict[str, Any] | None
    weekly_report: dict[str, Any] | None

    # ── Workflow control ─────────────────────────────────────────────────────
    current_week: int
    revision_cycle: int
    user_feedback: str | None
    hitl_action: str | None                        # approve|revise|end
    error_context: dict[str, Any] | None

    # ── Conversation ─────────────────────────────────────────────────────────
    # Uses LangGraph's built-in add_messages reducer for append semantics
    messages: Annotated[list[BaseMessage], add_messages]


def initial_state(
    user_id: str,
    target_role: str,
    session_id: str,
    github_profile_url: str | None = None,
    kaggle_username: str | None = None,
    uploaded_document_paths: list[str] | None = None,
) -> AgentState:
    """Return a fully-initialized AgentState with all required keys."""
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