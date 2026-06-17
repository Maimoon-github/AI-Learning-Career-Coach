from __future__ import annotations

import json
import tempfile
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.tools.ollama_tool import OllamaTool
from src.utils.llm_client import get_llm, get_embedder_config
from src.utils.error_handling import CrewExecutionError

log = structlog.get_logger(__name__)


class LLMFineTuningCrew:
    """Sequential crew: Data Preparer, Fine-tuning Orchestrator, Model Evaluator."""

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

        with open("config/agents.yaml") as f:
            agents_cfg = yaml.safe_load(f)["llm_fine_tuning"]
        with open("config/tasks.yaml") as f:
            self.tasks_cfg = yaml.safe_load(f)["fine_tuning"]

        llm = get_llm("structured_extraction")
        ollama_tool = OllamaTool()

        self.data_preparer = Agent(
            role=agents_cfg["data_preparer"]["role"],
            goal=agents_cfg["data_preparer"]["goal"],
            backstory=agents_cfg["data_preparer"]["backstory"],
            llm=llm,
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=4,
        )
        self.ft_orchestrator = Agent(
            role=agents_cfg["fine_tuning_orchestrator"]["role"],
            goal=agents_cfg["fine_tuning_orchestrator"]["goal"],
            backstory=agents_cfg["fine_tuning_orchestrator"]["backstory"],
            llm=llm,
            tools=[ollama_tool],
            verbose=False,
            allow_delegation=False,
            max_iter=5,
        )
        self.evaluator = Agent(
            role=agents_cfg["model_evaluator"]["role"],
            goal=agents_cfg["model_evaluator"]["goal"],
            backstory=agents_cfg["model_evaluator"]["backstory"],
            llm=llm,
            tools=[],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )

    def kickoff(self) -> dict[str, Any]:
        # Write notes to temp file for reference
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.raw_notes, f)
            notes_path = f.name

        data_task = Task(
            description=self.tasks_cfg["data_preparation"]["description"].format(
                raw_notes="\n---\n".join(self.raw_notes[:100])  # cap for prompt
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
            memory=True,
            embedder=get_embedder_config(),
        )
        log.info("fine_tuning_crew_starting", user_id=self.user_id, model=self.base_model)
        try:
            result = crew.kickoff(inputs={"user_id": self.user_id})
        except Exception as e:
            raise CrewExecutionError(f"LLMFineTuningCrew failed: {e}") from e
        finally:
            import os
            os.unlink(notes_path)

        raw = result.raw if hasattr(result, "raw") else str(result)
        try:
            output = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            output = {"status": "completed", "raw": raw}

        log.info("fine_tuning_crew_complete", user_id=self.user_id)
        return output if isinstance(output, dict) else {"status": "completed", "data": output}