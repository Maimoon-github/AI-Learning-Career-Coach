from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.utils.llm_client import get_llm
from src.utils.error_handling import CrewExecutionError

log = structlog.get_logger(__name__)


class ProgressReportingCrew:
    """Sequential crew: Data Aggregator, Report Generator, Motivational Coach."""

    def __init__(
        self,
        user_id: str,
        user_profile: dict[str, Any],
        week_number: int,
        raw_metrics: dict[str, Any],
    ) -> None:
        self.user_id = user_id
        self.user_profile = user_profile
        self.week_number = week_number
        self.raw_metrics = raw_metrics

        with open("config/agents.yaml") as f:
            agents_cfg = yaml.safe_load(f)["progress_reporting"]
        with open("config/tasks.yaml") as f:
            self.tasks_cfg = yaml.safe_load(f)["progress_reporting"]

        self.aggregator = Agent(
            role=agents_cfg["data_aggregator"]["role"],
            goal=agents_cfg["data_aggregator"]["goal"],
            backstory=agents_cfg["data_aggregator"]["backstory"],
            llm=get_llm("report_generation"),
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )
        self.report_writer = Agent(
            role=agents_cfg["report_generator"]["role"],
            goal=agents_cfg["report_generator"]["goal"],
            backstory=agents_cfg["report_generator"]["backstory"],
            llm=get_llm("report_generation"),
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )
        self.coach = Agent(
            role=agents_cfg["motivational_coach"]["role"],
            goal=agents_cfg["motivational_coach"]["goal"],
            backstory=agents_cfg["motivational_coach"]["backstory"],
            llm=get_llm("motivational_framing"),
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )

    def kickoff(self) -> dict[str, Any]:
        agg_task = Task(
            description=self.tasks_cfg["data_aggregation"]["description"].format(
                user_id=self.user_id, week_number=self.week_number
            ),
            expected_output=self.tasks_cfg["data_aggregation"]["expected_output"],
            agent=self.aggregator,
        )
        report_task = Task(
            description=self.tasks_cfg["report_generation"]["description"].format(
                week_number=self.week_number,
                metrics=json.dumps(self.raw_metrics, indent=2),
            ),
            expected_output=self.tasks_cfg["report_generation"]["expected_output"],
            agent=self.report_writer,
            context=[agg_task],
        )
        motivation_task = Task(
            description=self.tasks_cfg["motivational_framing"]["description"].format(
                draft_report="{report_task_output}",
                user_profile=json.dumps(self.user_profile, indent=2),
            ),
            expected_output=self.tasks_cfg["motivational_framing"]["expected_output"],
            agent=self.coach,
            context=[report_task],
        )

        crew = Crew(
            agents=[self.aggregator, self.report_writer, self.coach],
            tasks=[agg_task, report_task, motivation_task],
            process=Process.sequential,
            verbose=False,
        )
        log.info("progress_crew_starting", user_id=self.user_id, week=self.week_number)
        try:
            result = crew.kickoff(inputs={"user_id": self.user_id})
        except Exception as e:
            raise CrewExecutionError(f"ProgressReportingCrew failed: {e}") from e

        raw = result.raw if hasattr(result, "raw") else str(result)
        try:
            output = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            output = {"report": raw}

        log.info("progress_crew_complete", user_id=self.user_id)
        return output if isinstance(output, dict) else {"report": output}