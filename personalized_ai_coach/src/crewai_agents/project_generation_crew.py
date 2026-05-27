from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.models.project_model import ProjectSpec
from src.utils.llm_client import get_llm
from src.utils.error_handling import CrewExecutionError, ValidationError

log = structlog.get_logger(__name__)


class ProjectGenerationCrew:
    """Sequential crew: Project Ideator, Specification Writer, Difficulty Adjuster."""

    def __init__(self, skill_gap: dict, current_level: int, available_hours: float) -> None:
        self.skill_gap = skill_gap
        self.current_level = current_level
        self.available_hours = available_hours

        with open("config/agents.yaml") as f:
            agents_cfg = yaml.safe_load(f)["project_generation"]
        with open("config/tasks.yaml") as f:
            self.tasks_cfg = yaml.safe_load(f)["project_generation"]

        llm = get_llm("project_generation")

        self.ideator = Agent(
            role=agents_cfg["project_ideator"]["role"],
            goal=agents_cfg["project_ideator"]["goal"],
            backstory=agents_cfg["project_ideator"]["backstory"],
            llm=llm,
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )
        self.spec_writer = Agent(
            role=agents_cfg["specification_writer"]["role"],
            goal=agents_cfg["specification_writer"]["goal"],
            backstory=agents_cfg["specification_writer"]["backstory"],
            llm=llm,
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )
        self.difficulty_adjuster = Agent(
            role=agents_cfg["difficulty_adjuster"]["role"],
            goal=agents_cfg["difficulty_adjuster"]["goal"],
            backstory=agents_cfg["difficulty_adjuster"]["backstory"],
            llm=llm,
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )

    def kickoff(self) -> list[dict[str, Any]]:
        ideation_task = Task(
            description=self.tasks_cfg["project_ideation"]["description"].format(
                skill_gap=json.dumps(self.skill_gap),
                current_level=self.current_level,
                available_hours=self.available_hours,
            ),
            expected_output=self.tasks_cfg["project_ideation"]["expected_output"],
            agent=self.ideator,
        )
        spec_task = Task(
            description=self.tasks_cfg["specification_writing"]["description"].format(
                selected_project="{ideation_task_output}"
            ),
            expected_output=self.tasks_cfg["specification_writing"]["expected_output"],
            agent=self.spec_writer,
            context=[ideation_task],
        )
        calibration_task = Task(
            description=self.tasks_cfg["difficulty_calibration"]["description"].format(
                project_spec="{spec_task_output}",
                current_level=self.current_level,
            ),
            expected_output=self.tasks_cfg["difficulty_calibration"]["expected_output"],
            agent=self.difficulty_adjuster,
            context=[spec_task],
        )

        crew = Crew(
            agents=[self.ideator, self.spec_writer, self.difficulty_adjuster],
            tasks=[ideation_task, spec_task, calibration_task],
            process=Process.sequential,
            verbose=False,
        )
        log.info("project_crew_starting", skill=self.skill_gap.get("skill_name", ""))
        try:
            result = crew.kickoff()
        except Exception as e:
            raise CrewExecutionError(f"ProjectGenerationCrew failed: {e}") from e

        raw = result.raw if hasattr(result, "raw") else str(result)
        try:
            projects = json.loads(raw)
        except json.JSONDecodeError:
            projects = [{"raw_output": raw}]

        projects = projects if isinstance(projects, list) else [projects]
        validated = []
        for proj in projects:
            try:
                validated.append(ProjectSpec.model_validate(proj).model_dump())
            except Exception as e:
                log.warning("project_validation_failed", error=str(e))
        log.info("project_crew_complete", projects_generated=len(validated))
        return validated