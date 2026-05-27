from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.crewai_agents.skill_gap_assessment_crew import SkillGapAssessmentCrew
from src.langgraph_workflow.state import AgentState
from src.models.learning_path_model import SkillGap
from src.utils.error_handling import CrewExecutionError, async_retry_with_backoff, ValidationError

log = structlog.get_logger(__name__)


@async_retry_with_backoff(max_attempts=3, exceptions=(CrewExecutionError, Exception))
async def _run_crew(state: AgentState) -> list[dict]:
    crew = SkillGapAssessmentCrew(
        skill_profile=state["skill_profile"],
        target_role=state["target_role"],
    )
    gaps = await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)
    # Validate each gap against Pydantic schema
    validated = []
    for g in gaps:
        try:
            validated.append(SkillGap.model_validate(g).model_dump())
        except Exception as e:
            raise ValidationError(f"Invalid gap schema: {e}") from e
    return validated


async def skill_assessment_node(state: AgentState) -> dict[str, Any]:
    log.info("node.skill_assessment.start", user_id=state["user_id"])
    if not state.get("skill_profile"):
        return {"error_context": {"node": "skill_assessment", "error": "skill_profile is None"}}

    try:
        gaps = await _run_crew(state)
        return {"skill_gaps": gaps, "error_context": None}
    except Exception as exc:
        log.error("node.skill_assessment.error", error=str(exc))
        return {"error_context": {"node": "skill_assessment", "error": str(exc)}}