"""TypedDict state and Pydantic models."""

# src/state/schema.py

from __future__ import annotations
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


# ─────────────────────────────────────────────
# Sub-models (Pydantic for validation at edges)
# ─────────────────────────────────────────────

class UserProfile(BaseModel):
    """Persisted user profile — read from long-term memory at session start."""
    user_id: str
    name: str
    target_role: str                        # e.g. "ML Engineer", "Data Scientist"
    current_skills: list[str] = []
    skill_gaps: list[str] = []
    completed_topics: list[str] = []
    learning_velocity: float = 1.0          # multiplier: 0.5 slow, 1.0 avg, 2.0 fast
    preferred_resources: list[str] = []     # "video", "paper", "hands-on"
    github_username: Optional[str] = None
    kaggle_username: Optional[str] = None
    session_count: int = 0
    last_session: Optional[datetime] = None


class LearningPlan(BaseModel):
    """Structured learning plan produced by Curriculum Planner."""
    weeks: list[dict[str, Any]] = []        # [{week: 1, topic: "...", resources: [...]}]
    current_week: int = 1
    milestones: list[str] = []
    estimated_completion_days: int = 90


class PracticeProject(BaseModel):
    """A generated practice project from Project Builder."""
    title: str
    description: str
    difficulty: str                          # "beginner", "intermediate", "advanced"
    tech_stack: list[str]
    starter_code_outline: str
    evaluation_criteria: list[str]
    estimated_hours: int


class EvaluationResult(BaseModel):
    """LLM-as-Judge evaluation output."""
    score: float                             # 0.0 – 1.0
    passed: bool
    feedback: str
    retry_recommended: bool
    escalate_to_human: bool = False


class AgentHandover(BaseModel):
    """Structured handover schema between agents."""
    from_agent: str
    to_agent: str
    summary: str                             # What was accomplished
    context_payload: dict[str, Any]          # Data to pass forward
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Main State Object
# ─────────────────────────────────────────────

class CoachState(TypedDict):
    # ── Conversation ──────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]   # LangGraph reducer
    user_input: str
    voice_mode: bool

    # ── User context ──────────────────────────
    user_profile: UserProfile
    session_id: str
    is_new_user: bool

    # ── Agent outputs ─────────────────────────
    learning_plan: Optional[LearningPlan]
    current_project: Optional[PracticeProject]
    evaluation: Optional[EvaluationResult]
    weekly_report: Optional[str]

    # ── Routing & control ─────────────────────
    next_agent: str                          # Supervisor sets this
    iteration_count: int                     # Cycle guard
    max_iterations: int                      # Default: 5

    # ── RAG context ───────────────────────────
    retrieved_docs: list[dict[str, Any]]
    rag_query: str

    # ── HITL ──────────────────────────────────
    hitl_required: bool
    hitl_prompt: str
    human_approval: Optional[bool]

    # ── Handovers ─────────────────────────────
    handover_log: list[AgentHandover]

    # ── Fine-tuning signals ───────────────────
    new_notes: list[str]                    # User notes collected this session
    finetune_trigger: bool                  # Set True when enough data accumulates

    # ── Error tracking ────────────────────────
    error_count: int
    last_error: Optional[str]
