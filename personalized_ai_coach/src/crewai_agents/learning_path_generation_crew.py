from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.models.learning_path_model import LearningPath
from src.tools.web_search_tool import WebSearchTool
from src.utils.llm_client import get_llm
from src.utils.error_handling import CrewExecutionError, ValidationError

log = structlog.get_logger(__name__)


class LearningPathGenerationCrew:
    """Sequential crew: Curriculum Designer, Resource Curator, Path Optimizer."""

    def __init__(
        self,
        skill_gaps: list[dict],
        duration_weeks: int = 12,
        hours_per_week: int = 10,
    ) -> None:
        self.skill_gaps = skill_gaps
        self.duration_weeks = duration_weeks
        self.hours_per_week = hours_per_week

        with open("config/agents.yaml") as f:
            agents_cfg = yaml.safe_load(f)["learning_path"]
        with open("config/tasks.yaml") as f:
            self.tasks_cfg = yaml.safe_load(f)["learning_path"]

        self.curriculum_designer = Agent(
            role=agents_cfg["curriculum_designer"]["role"],
            goal=agents_cfg["curriculum_designer"]["goal"],
            backstory=agents_cfg["curriculum_designer"]["backstory"],
            llm=get_llm("curriculum_design"),
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=5,
        )
        self.resource_curator = Agent(
            role=agents_cfg["resource_curator"]["role"],
            goal=agents_cfg["resource_curator"]["goal"],
            backstory=agents_cfg["resource_curator"]["backstory"],
            llm=get_llm("resource_curation"),
            tools=[WebSearchTool()],
            verbose=False,
            allow_delegation=False,
            max_iter=6,
        )
        self.path_optimizer = Agent(
            role=agents_cfg["path_optimizer"]["role"],
            goal=agents_cfg["path_optimizer"]["goal"],
            backstory=agents_cfg["path_optimizer"]["backstory"],
            llm=get_llm("curriculum_design"),
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )

    def kickoff(self, user_feedback: str | None = None) -> dict[str, Any]:
        gaps_str = json.dumps(self.skill_gaps, indent=2)
        feedback_clause = f"\nUser feedback on previous path: {user_feedback}" if user_feedback else ""

        curriculum_task = Task(
            description=self.tasks_cfg["curriculum_design"]["description"].format(
                duration_weeks=self.duration_weeks,
                skill_gaps=gaps_str + feedback_clause,
                hours_per_week=self.hours_per_week,
            ),
            expected_output=self.tasks_cfg["curriculum_design"]["expected_output"],
            agent=self.curriculum_designer,
        )
        resource_task = Task(
            description=self.tasks_cfg["resource_curation"]["description"].format(
                curriculum="{curriculum_task_output}"
            ),
            expected_output=self.tasks_cfg["resource_curation"]["expected_output"],
            agent=self.resource_curator,
            context=[curriculum_task],
        )
        optimize_task = Task(
            description=self.tasks_cfg["path_optimization"]["description"].format(
                draft_path="{curriculum_task_output}",
                resources="{resource_task_output}",
                total_available_hours=self.duration_weeks * self.hours_per_week,
                max_weekly_hours=self.hours_per_week,
            ),
            expected_output=self.tasks_cfg["path_optimization"]["expected_output"],
            agent=self.path_optimizer,
            context=[curriculum_task, resource_task],
        )

        crew = Crew(
            agents=[self.curriculum_designer, self.resource_curator, self.path_optimizer],
            tasks=[curriculum_task, resource_task, optimize_task],
            process=Process.sequential,
            verbose=False,
            memory=False,  # memory=True requires OpenAI embeddings; incompatible with Ollama-only setup
        )
        log.info("learning_path_crew_starting", weeks=self.duration_weeks)
        try:
            result = crew.kickoff()
        except Exception as e:
            raise CrewExecutionError(f"LearningPathGenerationCrew failed: {e}") from e

        raw = result.raw if hasattr(result, "raw") else str(result)
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            output = {"path": raw}

        try:
            LearningPath.model_validate(output)
        except Exception as e:
            raise ValidationError(f"Learning path output does not match schema: {e}") from e

        log.info("learning_path_crew_complete")
        return output