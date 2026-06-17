from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.crewai_agents.profile_analysis_crew import ProfileAnalysisCrew
from src.langgraph_workflow.state import AgentState
from src.utils.error_handling import CrewExecutionError, async_retry_with_backoff

log = structlog.get_logger(__name__)


@async_retry_with_backoff(max_attempts=3, exceptions=(CrewExecutionError, Exception))
async def _run_crew(state: AgentState) -> dict:
    crew = ProfileAnalysisCrew(
        user_id=state["user_id"],
        github_url=state.get("github_profile_url"),
        kaggle_username=state.get("kaggle_username"),
        document_paths=state.get("uploaded_document_paths", []),
    )
    return await asyncio.get_running_loop().run_in_executor(None, crew.kickoff)


async def profile_ingestion_node(state: AgentState) -> dict[str, Any]:
    log.info("node.profile_ingestion.start", user_id=state["user_id"])
    try:
        profile = await _run_crew(state)
        return {"skill_profile": profile, "error_context": None}
    except Exception as exc:
        log.error("node.profile_ingestion.error", error=str(exc))
        return {"error_context": {"node": "profile_ingestion", "error": str(exc)}}