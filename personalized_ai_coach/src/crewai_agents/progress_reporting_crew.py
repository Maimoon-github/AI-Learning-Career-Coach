from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.tools.tools import OllamaTool
from src.utils.llm_client import get_llm

log = structlog.get_logger(__name__)


def _cfg(section: str) -> tuple[dict, dict]:
    with open("config/agents.yaml") as f:
        a = yaml.safe_load(f)[section]
    with open("config/tasks.yaml") as f:
        t = yaml.safe_load(f)[section]
    return a, t


def _parse(result: Any) -> dict | list:
    raw = result.raw if hasattr(result, "raw") else str(result)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"raw_output": str(raw)}


# ── LLM Fine-Tuning Crew ──────────────────────────────────────────────────────

class LLMFineTuningCrew:
    def __init__(
        self,
        user_id: str,
        raw_notes: list[str],
        base_model: str = "llama3.2:3b",
        epochs: int = 3,
        lora_rank: int = 16,
        learning_rate: float = 0.0002,
    ) -> None:
        self.user_id = user_id
        self.raw_notes = raw_notes
        self.base_model = base_model
        self.epochs = epochs
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate

        agents_cfg, self.tasks_cfg = _cfg("llm_fine_tuning")
        llm = get_llm("structured_extraction")
        ollama_tool = OllamaTool()

        self.data_preparer = Agent(
            role=agents_cfg["data_preparer"]["role"],
            goal=agents_cfg["data_preparer"]["goal"],
            backstory=agents_cfg["data_preparer"]["backstory"],
            llm=llm, tools=[], verbose=False, allow_delegation=False, max_iter=4,
        )
        self.ft_orchestrator = Agent(
            role=agents_cfg["fine_tuning_orchestrator"]["role"],
            goal=agents_cfg["fine_tuning_orchestrator"]["goal"],
            backstory=agents_cfg["fine_tuning_orchestrator"]["backstory"],
            llm=llm, tools=[ollama_tool], verbose=False, allow_delegation=False, max_iter=5,
        )
        self.evaluator = Agent(
            role=agents_cfg["model_evaluator"]["role"],
            goal=agents_cfg["model_evaluator"]["goal"],
            backstory=agents_cfg["model_evaluator"]["backstory"],
            llm=llm, tools=[], verbose=False, allow_delegation=False, max_iter=3,
        )

    def kickoff(self) -> dict[str, Any]:
        import os, tempfile, json as _json

        # Write notes to a temp file for the crew to reference
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            _json.dump(self.raw_notes, f)
            notes_path = f.name

        data_task = Task(
            description=self.tasks_cfg["data_preparation"]["description"].format(
                raw_notes="\n---\n".join(self.raw_notes[:100])  # Cap at 100 for prompt size
            ),
            expected_output=self.tasks_cfg["data_preparation"]["expected_output"],
            agent=self.data_preparer,
        )
        ft_task = Task(
            description=self.tasks_cfg["fine_tuning_execution"]["description"].format(
                base_model=self.base_model,
                data_path=notes_path,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                lora_rank=self.lora_rank,
            ),
            expected_output=self.tasks_cfg["fine_tuning_execution"]["expected_output"],
            agent=self.ft_orchestrator,
            context=[data_task],
        )
        eval_task = Task(
            description=self.tasks_cfg["model_evaluation"]["description"].format(
                fine_tuned_model=f"{self.base_model}-{self.user_id}-ft",
                base_model=self.base_model,
                eval_prompts="[standard personalization eval set]",
            ),
            expected_output=self.tasks_cfg["model_evaluation"]["expected_output"],
            agent=self.evaluator,
            context=[ft_task],
        )

        crew = Crew(
            agents=[self.data_preparer, self.ft_orchestrator, self.evaluator],
            tasks=[data_task, ft_task, eval_task],
            process=Process.sequential,
            verbose=False,
        )
        log.info("fine_tuning_crew_starting", user_id=self.user_id, model=self.base_model)
        result = _parse(crew.kickoff(inputs={"user_id": self.user_id}))
        log.info("fine_tuning_crew_complete", user_id=self.user_id)
        os.unlink(notes_path)
        return result if isinstance(result, dict) else {"status": "complete", "raw": result}


# ── Progress Reporting Crew ───────────────────────────────────────────────────

class ProgressReportingCrew:
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

        agents_cfg, self.tasks_cfg = _cfg("progress_reporting")
        llm = get_llm("report_generation")
        creative_llm = get_llm("motivational_framing")

        self.aggregator = Agent(
            role=agents_cfg["data_aggregator"]["role"],
            goal=agents_cfg["data_aggregator"]["goal"],
            backstory=agents_cfg["data_aggregator"]["backstory"],
            llm=llm, tools=[], verbose=False, allow_delegation=False, max_iter=3,
        )
        self.report_writer = Agent(
            role=agents_cfg["report_generator"]["role"],
            goal=agents_cfg["report_generator"]["goal"],
            backstory=agents_cfg["report_generator"]["backstory"],
            llm=llm, tools=[], verbose=False, allow_delegation=False, max_iter=3,
        )
        self.coach = Agent(
            role=agents_cfg["motivational_coach"]["role"],
            goal=agents_cfg["motivational_coach"]["goal"],
            backstory=agents_cfg["motivational_coach"]["backstory"],
            llm=creative_llm, tools=[], verbose=False, allow_delegation=False, max_iter=3,
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
        result = _parse(crew.kickoff(inputs={"user_id": self.user_id}))
        log.info("progress_crew_complete", user_id=self.user_id)
        return result if isinstance(result, dict) else {"report": result}