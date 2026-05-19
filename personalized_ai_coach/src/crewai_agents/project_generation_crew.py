from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.tools.tools import WebSearchTool
from src.utils.llm_client import get_llm

log = structlog.get_logger(__name__)


def _cfg(section: str) -> tuple[dict, dict]:
    with open("config/agents.yaml") as f:
        a = yaml.safe_load(f)[section]
    with open("config/tasks.yaml") as f:
        t = yaml.safe_load(f)[section]
    return a, t


def _parse_result(result: Any) -> dict | list:
    raw = result.raw if hasattr(result, "raw") else str(result)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"raw_output": str(raw)}


# ── Skill Gap Assessment Crew ─────────────────────────────────────────────────

class SkillGapAssessmentCrew:
    def __init__(self, skill_profile: dict[str, Any], target_role: str) -> None:
        self.skill_profile = skill_profile
        self.target_role = target_role
        agents_cfg, self.tasks_cfg = _cfg("skill_gap_assessment")
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
        result = _parse_result(crew.kickoff(inputs={"target_role": self.target_role}))
        gaps = result if isinstance(result, list) else result.get("gaps", [result])
        log.info("skill_gap_crew_complete", gaps_found=len(gaps))
        return gaps


# ── Learning Path Generation Crew ─────────────────────────────────────────────

class LearningPathGenerationCrew:
    def __init__(
        self,
        skill_gaps: list[dict],
        duration_weeks: int = 12,
        hours_per_week: int = 10,
    ) -> None:
        self.skill_gaps = skill_gaps
        self.duration_weeks = duration_weeks
        self.hours_per_week = hours_per_week
        agents_cfg, self.tasks_cfg = _cfg("learning_path")
        search_tool = WebSearchTool()

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
            tools=[search_tool],
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
            memory=True,
        )
        log.info("learning_path_crew_starting", weeks=self.duration_weeks)
        result = _parse_result(crew.kickoff())
        log.info("learning_path_crew_complete")
        return result if isinstance(result, dict) else {"path": result}


# ── Project Generation Crew ───────────────────────────────────────────────────

class ProjectGenerationCrew:
    def __init__(self, skill_gap: dict, current_level: int, available_hours: float) -> None:
        self.skill_gap = skill_gap
        self.current_level = current_level
        self.available_hours = available_hours
        agents_cfg, self.tasks_cfg = _cfg("project_generation")
        llm = get_llm("project_generation")

        self.ideator = Agent(
            role=agents_cfg["project_ideator"]["role"],
            goal=agents_cfg["project_ideator"]["goal"],
            backstory=agents_cfg["project_ideator"]["backstory"],
            llm=llm, tools=[], verbose=False, allow_delegation=False, max_iter=3,
        )
        self.spec_writer = Agent(
            role=agents_cfg["specification_writer"]["role"],
            goal=agents_cfg["specification_writer"]["goal"],
            backstory=agents_cfg["specification_writer"]["backstory"],
            llm=llm, tools=[], verbose=False, allow_delegation=False, max_iter=3,
        )
        self.difficulty_adjuster = Agent(
            role=agents_cfg["difficulty_adjuster"]["role"],
            goal=agents_cfg["difficulty_adjuster"]["goal"],
            backstory=agents_cfg["difficulty_adjuster"]["backstory"],
            llm=llm, tools=[], verbose=False, allow_delegation=False, max_iter=3,
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
        result = _parse_result(crew.kickoff())
        projects = result if isinstance(result, list) else [result]
        log.info("project_crew_complete", projects_generated=len(projects))
        return projects