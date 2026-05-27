from __future__ import annotations

import asyncio
from typing import Any

import structlog
import yaml

from src.crewai_agents.learning_path_generation_crew import LearningPathGenerationCrew
from src.langgraph_workflow.state import AgentState
from src.models.learning_path_model import LearningPath
from src.utils.error_handling import CrewExecutionError, async_retry_with_backoff, ValidationError

log = structlog.get_logger(__name__)


def _load_settings() -> dict:
    with open("config/system_settings.yaml") as f:
        return yaml.safe_load(f)


@async_retry_with_backoff(max_attempts=3, exceptions=(CrewExecutionError, Exception))
async def _run_crew(state: AgentState) -> dict:
    settings = _load_settings()
    crew = LearningPathGenerationCrew(
        skill_gaps=state.get("skill_gaps", []),
        duration_weeks=settings["learning"]["default_duration_weeks"],
        hours_per_week=settings["learning"]["default_hours_per_week"],
    )
    path = await asyncio.get_event_loop().run_in_executor(
        None, lambda: crew.kickoff(user_feedback=state.get("user_feedback"))
    )
    # Validate against Pydantic model
    LearningPath.model_validate(path)
    return path


async def learning_path_node(state: AgentState) -> dict[str, Any]:
    log.info("node.learning_path.start", user_id=state["user_id"])
    try:
        path = await _run_crew(state)
        return {"learning_path": path, "user_feedback": None, "error_context": None}
    except Exception as exc:
        log.error("node.learning_path.error", error=str(exc))
        return {"error_context": {"node": "learning_path", "error": str(exc)}}