from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog
import yaml

from src.crewai_agents.crews import (
    LearningPathGenerationCrew,
    ProjectGenerationCrew,
    SkillGapAssessmentCrew,
)
from src.crewai_agents.profile_analysis_crew import ProfileAnalysisCrew
from src.crewai_agents.specialized_crews import LLMFineTuningCrew, ProgressReportingCrew
from src.langgraph_workflow.state import AgentState
from src.utils.error_handling import CrewExecutionError, HITLTimeoutError, with_retry

log = structlog.get_logger(__name__)


def _load_settings() -> dict:
    with open("config/system_settings.yaml") as f:
        return yaml.safe_load(f)


# ── Profile Ingestion Node ────────────────────────────────────────────────────

async def profile_ingestion_node(state: AgentState) -> dict[str, Any]:
    """Invoke ProfileAnalysisCrew to build the user's skill profile."""
    log.info("node.profile_ingestion.start", user_id=state["user_id"])

    @with_retry(max_attempts=3, retriable_errors=(CrewExecutionError, Exception))
    async def _run() -> dict:
        crew = ProfileAnalysisCrew(
            user_id=state["user_id"],
            github_url=state.get("github_profile_url"),
            kaggle_username=state.get("kaggle_username"),
            document_paths=state.get("uploaded_document_paths", []),
        )
        return await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

    try:
        profile = await _run()
        log.info("node.profile_ingestion.complete", user_id=state["user_id"])
        return {"skill_profile": profile, "error_context": None}
    except Exception as exc:
        log.error("node.profile_ingestion.error", error=str(exc))
        return {"error_context": {"node": "profile_ingestion", "error": str(exc)}}


# ── Skill Assessment Node ─────────────────────────────────────────────────────

async def skill_assessment_node(state: AgentState) -> dict[str, Any]:
    """Invoke SkillGapAssessmentCrew against the user's skill profile."""
    log.info("node.skill_assessment.start", user_id=state["user_id"])

    if not state.get("skill_profile"):
        log.error("node.skill_assessment.missing_profile")
        return {"error_context": {"node": "skill_assessment", "error": "skill_profile is None"}}

    @with_retry(max_attempts=3, retriable_errors=(CrewExecutionError, Exception))
    async def _run() -> list:
        crew = SkillGapAssessmentCrew(
            skill_profile=state["skill_profile"],
            target_role=state["target_role"],
        )
        return await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

    try:
        gaps = await _run()
        log.info("node.skill_assessment.complete", gaps_count=len(gaps))
        return {"skill_gaps": gaps, "error_context": None}
    except Exception as exc:
        log.error("node.skill_assessment.error", error=str(exc))
        return {"error_context": {"node": "skill_assessment", "error": str(exc)}}


# ── Learning Path Node ────────────────────────────────────────────────────────

async def learning_path_node(state: AgentState) -> dict[str, Any]:
    """Generate or update the learning path, incorporating user feedback on revisions."""
    log.info("node.learning_path.start", user_id=state["user_id"])
    settings = _load_settings()

    @with_retry(max_attempts=3, retriable_errors=(CrewExecutionError, Exception))
    async def _run() -> dict:
        crew = LearningPathGenerationCrew(
            skill_gaps=state.get("skill_gaps", []),
            duration_weeks=settings["learning"]["default_duration_weeks"],
            hours_per_week=settings["learning"]["default_hours_per_week"],
        )
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: crew.kickoff(user_feedback=state.get("user_feedback")),
        )

    try:
        path = await _run()
        log.info("node.learning_path.complete", user_id=state["user_id"])
        return {
            "learning_path": path,
            "user_feedback": None,   # Clear feedback after processing
            "error_context": None,
        }
    except Exception as exc:
        log.error("node.learning_path.error", error=str(exc))
        return {"error_context": {"node": "learning_path", "error": str(exc)}}


# ── Project Generation Node ───────────────────────────────────────────────────

async def project_generation_node(state: AgentState) -> dict[str, Any]:
    """Generate practice projects for the top-priority skill gaps concurrently."""
    log.info("node.project_generation.start", user_id=state["user_id"])
    gaps = state.get("skill_gaps", [])
    # Process top 3 highest-severity gaps in parallel
    top_gaps = sorted(gaps, key=lambda g: g.get("gap_severity", 0), reverse=True)[:3]

    if not top_gaps:
        return {"practice_projects": [], "error_context": None}

    async def _generate_for_gap(gap: dict) -> list[dict]:
        crew = ProjectGenerationCrew(
            skill_gap=gap,
            current_level=gap.get("current_level", 1),
            available_hours=gap.get("weeks_to_close", 2) * 10,
        )
        return await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

    try:
        results = await asyncio.gather(*[_generate_for_gap(g) for g in top_gaps])
        all_projects = [p for projects in results for p in projects]
        log.info("node.project_generation.complete", projects_count=len(all_projects))
        return {"practice_projects": all_projects, "error_context": None}
    except Exception as exc:
        log.error("node.project_generation.error", error=str(exc))
        return {"error_context": {"node": "project_generation", "error": str(exc)}}


# ── LLM Fine-Tuning Node ──────────────────────────────────────────────────────

async def llm_fine_tuning_node(state: AgentState) -> dict[str, Any]:
    """Trigger fine-tuning if sufficient session notes exist."""
    log.info("node.llm_fine_tuning.start", user_id=state["user_id"])
    settings = _load_settings()
    notes = state.get("session_notes", [])

    if len(notes) < settings["fine_tuning"]["min_examples_required"]:
        log.info(
            "node.llm_fine_tuning.skipped",
            reason="insufficient_notes",
            count=len(notes),
            required=settings["fine_tuning"]["min_examples_required"],
        )
        return {"fine_tuning_status": "skipped", "fine_tuning_metrics": None}

    @with_retry(max_attempts=2, retriable_errors=(Exception,))
    async def _run() -> dict:
        crew = LLMFineTuningCrew(
            user_id=state["user_id"],
            raw_notes=notes,
            epochs=settings["fine_tuning"]["default_epochs"],
            lora_rank=settings["fine_tuning"]["default_lora_rank"],
            learning_rate=settings["fine_tuning"]["default_learning_rate"],
        )
        return await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

    try:
        metrics = await _run()
        log.info("node.llm_fine_tuning.complete", user_id=state["user_id"])
        return {"fine_tuning_status": "complete", "fine_tuning_metrics": metrics, "error_context": None}
    except Exception as exc:
        log.error("node.llm_fine_tuning.error", error=str(exc))
        return {
            "fine_tuning_status": "failed",
            "error_context": {"node": "llm_fine_tuning", "error": str(exc)},
        }


# ── Progress Report Node ──────────────────────────────────────────────────────

async def progress_report_node(state: AgentState) -> dict[str, Any]:
    """Aggregate metrics and produce the weekly progress report."""
    log.info("node.progress_report.start", user_id=state["user_id"], week=state.get("current_week", 1))

    # Compute raw metrics from state
    learning_path = state.get("learning_path") or {}
    weeks_data = learning_path.get("weeks", [])
    current_week = state.get("current_week", 1)
    week_data = weeks_data[current_week - 1] if weeks_data and current_week <= len(weeks_data) else {}

    raw_metrics = {
        "week_number": current_week,
        "planned_topics": week_data.get("topics", []),
        "planned_hours": week_data.get("estimated_hours", 0),
        "projects_completed": len([
            p for p in state.get("practice_projects", [])
            if p.get("status") == "completed"
        ]),
        "fine_tuning_status": state.get("fine_tuning_status"),
        "skill_gaps_remaining": len(state.get("skill_gaps", [])),
    }

    @with_retry(max_attempts=2, retriable_errors=(Exception,))
    async def _run() -> dict:
        crew = ProgressReportingCrew(
            user_id=state["user_id"],
            user_profile=state.get("skill_profile") or {},
            week_number=current_week,
            raw_metrics=raw_metrics,
        )
        return await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)

    try:
        report = await _run()
        log.info("node.progress_report.complete", user_id=state["user_id"])
        return {"weekly_report": report, "error_context": None}
    except Exception as exc:
        log.error("node.progress_report.error", error=str(exc))
        return {"error_context": {"node": "progress_report", "error": str(exc)}}


# ── HITL Node ─────────────────────────────────────────────────────────────────

async def hitl_node(state: AgentState) -> dict[str, Any]:
    """
    Human-in-the-Loop breakpoint. In production this suspends via LangGraph
    interrupt() and resumes when the user submits approve/revise/end.
    In this implementation we model the interrupt contract directly.
    """
    from langgraph.types import interrupt

    log.info("node.hitl.waiting", user_id=state["user_id"], week=state.get("current_week"))
    settings = _load_settings()

    presentation = {
        "weekly_report": state.get("weekly_report"),
        "learning_path_preview": (state.get("learning_path") or {}).get("weeks", [])[:2],
        "projects_preview": state.get("practice_projects", [])[:3],
        "current_week": state.get("current_week", 1),
        "revision_cycle": state.get("revision_cycle", 0),
    }

    # LangGraph interrupt() suspends execution and returns when resumed with user input
    user_response = interrupt(value=presentation)

    # user_response expected: {"action": "approve"|"revise"|"end", "feedback": str|None}
    action = user_response.get("action", "approve") if isinstance(user_response, dict) else "approve"
    feedback = user_response.get("feedback") if isinstance(user_response, dict) else None

    log.info("node.hitl.resumed", user_id=state["user_id"], action=action)

    return {
        "hitl_action": action,
        "user_feedback": feedback,
        "revision_cycle": state.get("revision_cycle", 0) + (1 if action == "revise" else 0),
    }