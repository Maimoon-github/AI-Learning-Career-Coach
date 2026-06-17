from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.crewai_agents.progress_reporting_crew import ProgressReportingCrew
from src.langgraph_workflow.state import AgentState
from src.utils.error_handling import CrewExecutionError, async_retry_with_backoff

log = structlog.get_logger(__name__)


def _compute_raw_metrics(state: AgentState) -> dict:
    learning_path = state.get("learning_path") or {}
    weeks = learning_path.get("weeks", [])
    current_week = state.get("current_week", 1)
    week_data = weeks[current_week - 1] if weeks and current_week <= len(weeks) else {}

    return {
        "week_number": current_week,
        "planned_topics": week_data.get("topics", []),
        "planned_hours": week_data.get("estimated_hours", 0),
        "projects_completed": len([p for p in state.get("practice_projects", []) if p.get("status") == "completed"]),
        "fine_tuning_status": state.get("fine_tuning_status"),
        "skill_gaps_remaining": len(state.get("skill_gaps", [])),
    }


@async_retry_with_backoff(max_attempts=2, exceptions=(CrewExecutionError, Exception))
async def _run_crew(state: AgentState, raw_metrics: dict) -> dict:
    crew = ProgressReportingCrew(
        user_id=state["user_id"],
        user_profile=state.get("skill_profile") or {},
        week_number=state.get("current_week", 1),
        raw_metrics=raw_metrics,
    )
    return await asyncio.get_running_loop().run_in_executor(None, crew.kickoff)


async def progress_report_node(state: AgentState) -> dict[str, Any]:
    log.info("node.progress_report.start", user_id=state["user_id"], week=state.get("current_week", 1))
    raw_metrics = _compute_raw_metrics(state)

    try:
        report = await _run_crew(state, raw_metrics)
        return {"weekly_report": report, "error_context": None}
    except Exception as exc:
        log.error("node.progress_report.error", error=str(exc))
        return {"error_context": {"node": "progress_report", "error": str(exc)}}