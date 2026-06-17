from __future__ import annotations

import asyncio
from typing import Any

import structlog
import yaml

from src.crewai_agents.llm_fine_tuning_crew import LLMFineTuningCrew
from src.langgraph_workflow.state import AgentState
from src.utils.error_handling import CrewExecutionError, async_retry_with_backoff

log = structlog.get_logger(__name__)


def _load_settings() -> dict:
    with open("config/system_settings.yaml") as f:
        return yaml.safe_load(f)


@async_retry_with_backoff(max_attempts=2, exceptions=(CrewExecutionError, Exception))
async def _run_crew(state: AgentState) -> dict:
    settings = _load_settings()
    crew = LLMFineTuningCrew(
        user_id=state["user_id"],
        raw_notes=state.get("session_notes", []),
        epochs=settings["fine_tuning"]["default_epochs"],
        lora_rank=settings["fine_tuning"]["default_lora_rank"],
        learning_rate=settings["fine_tuning"]["default_learning_rate"],
    )
    return await asyncio.get_running_loop().run_in_executor(None, crew.kickoff)


async def llm_fine_tuning_node(state: AgentState) -> dict[str, Any]:
    log.info("node.llm_fine_tuning.start", user_id=state["user_id"])
    settings = _load_settings()
    notes = state.get("session_notes", [])
    min_required = settings["fine_tuning"]["min_examples_required"]

    if len(notes) < min_required:
        log.info("node.llm_fine_tuning.skipped", count=len(notes), required=min_required)
        return {"fine_tuning_status": "skipped", "fine_tuning_metrics": None, "error_context": None}

    try:
        metrics = await _run_crew(state)
        return {"fine_tuning_status": "complete", "fine_tuning_metrics": metrics, "error_context": None}
    except Exception as exc:
        log.error("node.llm_fine_tuning.error", error=str(exc))
        return {"fine_tuning_status": "failed", "error_context": {"node": "llm_fine_tuning", "error": str(exc)}}