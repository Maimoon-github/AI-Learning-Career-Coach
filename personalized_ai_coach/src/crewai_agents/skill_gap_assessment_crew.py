from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.models.learning_path_model import SkillGap
from src.tools.web_search_tool import WebSearchTool
from src.utils.llm_client import get_llm
from src.utils.error_handling import CrewExecutionError

log = structlog.get_logger(__name__)


class SkillGapAssessmentCrew:
    """Sequential crew: Role Definition Agent, Gap Analyst."""

    def __init__(self, skill_profile: dict[str, Any], target_role: str) -> None:
        self.skill_profile = skill_profile
        self.target_role = target_role

        with open("config/agents.yaml") as f:
            agents_cfg = yaml.safe_load(f)["skill_gap_assessment"]
        with open("config/tasks.yaml") as f:
            self.tasks_cfg = yaml.safe_load(f)["skill_gap"]

        llm = get_llm("gap_analysis")
        search_tool = WebSearchTool()

        self.role_agent = Agent(
            role=agents_cfg["role_definition_agent"]["role"],
            goal=agents_cfg["role_definition_agent"]["goal"],
            backstory=agents_cfg["role_definition_agent"]["backstory"],
            llm=llm,
            tools=[search_tool],
            verbose=False,
            allow_delegation=False,
            max_iter=4,
        )
        self.gap_analyst = Agent(
            role=agents_cfg["gap_analyst"]["role"],
            goal=agents_cfg["gap_analyst"]["goal"],
            backstory=agents_cfg["gap_analyst"]["backstory"],
            llm=llm,
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )

    def kickoff(self) -> list[dict[str, Any]]:
        role_task = Task(
            description=self.tasks_cfg["role_requirements_extraction"]["description"].format(
                target_role=self.target_role
            ),
            expected_output=self.tasks_cfg["role_requirements_extraction"]["expected_output"],
            agent=self.role_agent,
        )
        gap_task = Task(
            description=self.tasks_cfg["gap_analysis"]["description"].format(
                skill_profile=json.dumps(self.skill_profile, indent=2),
                role_requirements="{role_task_output}",
            ),
            expected_output=self.tasks_cfg["gap_analysis"]["expected_output"],
            agent=self.gap_analyst,
            context=[role_task],
        )
        crew = Crew(
            agents=[self.role_agent, self.gap_analyst],
            tasks=[role_task, gap_task],
            process=Process.sequential,
            verbose=False,
        )
        log.info("skill_gap_crew_starting", target_role=self.target_role)
        try:
            result = crew.kickoff(inputs={"target_role": self.target_role})
        except Exception as e:
            raise CrewExecutionError(f"SkillGapAssessmentCrew failed: {e}") from e

        raw = result.raw if hasattr(result, "raw") else str(result)
        try:
            gaps = json.loads(raw)
        except json.JSONDecodeError:
            gaps = [{"raw_output": raw}]

        # Validate each gap against SkillGap model
        validated = []
        for g in (gaps if isinstance(gaps, list) else [gaps]):
            try:
                validated.append(SkillGap.model_validate(g).model_dump())
            except Exception as e:
                log.warning("gap_validation_failed", error=str(e))
        log.info("skill_gap_crew_complete", gaps_found=len(validated))
        return validated