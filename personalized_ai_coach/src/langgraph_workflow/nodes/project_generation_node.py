from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.crewai_agents.project_generation_crew import ProjectGenerationCrew
from src.langgraph_workflow.state import AgentState
from src.models.project_model import ProjectSpec
from src.utils.error_handling import CrewExecutionError, async_retry_with_backoff, ValidationError

log = structlog.get_logger(__name__)


async def _generate_for_gap(gap: dict) -> list[dict]:
    crew = ProjectGenerationCrew(
        skill_gap=gap,
        current_level=gap.get("current_level", 1),
        available_hours=gap.get("weeks_to_close", 2) * 10,
    )
    projects = await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)
    validated = []
    for p in projects:
        try:
            validated.append(ProjectSpec.model_validate(p).model_dump())
        except Exception as e:
            raise ValidationError(f"Invalid project schema: {e}") from e
    return validated


async def project_generation_node(state: AgentState) -> dict[str, Any]:
    log.info("node.project_generation.start", user_id=state["user_id"])
    gaps = state.get("skill_gaps", [])
    top_gaps = sorted(gaps, key=lambda g: g.get("gap_severity", 0), reverse=True)[:3]

    if not top_gaps:
        return {"practice_projects": [], "error_context": None}

    try:
        results = await asyncio.gather(*[_generate_for_gap(g) for g in top_gaps])
        all_projects = [p for projects in results for p in projects]
        return {"practice_projects": all_projects, "error_context": None}
    except Exception as exc:
        log.error("node.project_generation.error", error=str(exc))
        return {"error_context": {"node": "project_generation", "error": str(exc)}}