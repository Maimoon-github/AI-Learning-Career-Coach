from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task

from src.models.skill_profile_model import SkillProfile
from src.tools.github_tool import GitHubTool
from src.tools.kaggle_tool import KaggleTool
from src.tools.document_parser_tool import DocumentParserTool
from src.utils.llm_client import get_llm
from src.utils.error_handling import CrewExecutionError, ValidationError

log = structlog.get_logger(__name__)


class ProfileAnalysisCrew:
    """Sequential crew: GitHub Analyst, Kaggle Analyst, Document Processor, Profile Synthesizer."""

    def __init__(
        self,
        user_id: str,
        github_url: str | None = None,
        kaggle_username: str | None = None,
        document_paths: list[str] | None = None,
    ) -> None:
        self.user_id = user_id
        self.github_url = github_url
        self.kaggle_username = kaggle_username
        self.document_paths = document_paths or []

        with open("config/agents.yaml") as f:
            agents_cfg = yaml.safe_load(f)["profile_analysis"]
        with open("config/tasks.yaml") as f:
            self.tasks_cfg = yaml.safe_load(f)["profile_analysis"]

        llm = get_llm("structured_extraction")

        self.github_analyst = Agent(
            role=agents_cfg["github_analyst"]["role"],
            goal=agents_cfg["github_analyst"]["goal"],
            backstory=agents_cfg["github_analyst"]["backstory"],
            llm=llm,
            tools=[GitHubTool()],
            verbose=agents_cfg["github_analyst"].get("verbose", False),
            allow_delegation=agents_cfg["github_analyst"].get("allow_delegation", False),
            max_iter=3,
        )
        self.kaggle_analyst = Agent(
            role=agents_cfg["kaggle_analyst"]["role"],
            goal=agents_cfg["kaggle_analyst"]["goal"],
            backstory=agents_cfg["kaggle_analyst"]["backstory"],
            llm=llm,
            tools=[KaggleTool()],
            verbose=agents_cfg["kaggle_analyst"].get("verbose", False),
            allow_delegation=agents_cfg["kaggle_analyst"].get("allow_delegation", False),
            max_iter=3,
        )
        self.doc_processor = Agent(
            role=agents_cfg["document_processor"]["role"],
            goal=agents_cfg["document_processor"]["goal"],
            backstory=agents_cfg["document_processor"]["backstory"],
            llm=llm,
            tools=[DocumentParserTool()],
            verbose=agents_cfg["document_processor"].get("verbose", False),
            allow_delegation=agents_cfg["document_processor"].get("allow_delegation", False),
            max_iter=3,
        )
        self.synthesizer = Agent(
            role=agents_cfg["profile_synthesizer"]["role"],
            goal=agents_cfg["profile_synthesizer"]["goal"],
            backstory=agents_cfg["profile_synthesizer"]["backstory"],
            llm=get_llm("structured_extraction"),
            tools=[],
            verbose=agents_cfg["profile_synthesizer"].get("verbose", False),
            allow_delegation=agents_cfg["profile_synthesizer"].get("allow_delegation", False),
            max_iter=5,
        )

    def _build_tasks(self) -> list[Task]:
        tasks = []
        if self.github_url:
            tasks.append(Task(
                description=self.tasks_cfg["github_analysis"]["description"].format(
                    github_url=self.github_url, user_id=self.user_id
                ),
                expected_output=self.tasks_cfg["github_analysis"]["expected_output"],
                agent=self.github_analyst,
            ))
        if self.kaggle_username:
            tasks.append(Task(
                description=self.tasks_cfg["kaggle_analysis"]["description"].format(
                    kaggle_url=f"https://kaggle.com/{self.kaggle_username}",
                    user_id=self.user_id,
                ),
                expected_output=self.tasks_cfg["kaggle_analysis"]["expected_output"],
                agent=self.kaggle_analyst,
            ))
        if self.document_paths:
            tasks.append(Task(
                description=self.tasks_cfg["document_processing"]["description"].format(
                    user_id=self.user_id,
                    document_paths=", ".join(self.document_paths),
                ),
                expected_output=self.tasks_cfg["document_processing"]["expected_output"],
                agent=self.doc_processor,
            ))
        # Synthesis task uses context from all previous tasks
        tasks.append(Task(
            description=self.tasks_cfg["profile_synthesis"]["description"],
            expected_output=self.tasks_cfg["profile_synthesis"]["expected_output"],
            agent=self.synthesizer,
            context=tasks,
        ))
        return tasks

    def kickoff(self) -> dict[str, Any]:
        tasks = self._build_tasks()
        active_agents = {t.agent for t in tasks}
        crew = Crew(
            agents=list(active_agents),
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            memory=False,  # Disabled: requires OpenAI embeddings by default
        )
        log.info("profile_crew_starting", user_id=self.user_id)
        try:
            result = crew.kickoff(inputs={"user_id": self.user_id})
        except Exception as e:
            raise CrewExecutionError(f"ProfileAnalysisCrew failed: {e}") from e

        raw = result.raw if hasattr(result, "raw") else str(result)
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            output = {"raw_output": raw}

        # Unwrap nested schema formats like {"name": "...", "parameters": {...}}
        if "parameters" in output and isinstance(output.get("parameters"), dict):
            output = output["parameters"]

        # Inject required fields that the LLM is not expected to generate
        output.setdefault("user_id", self.user_id)
        output.setdefault("target_role", "Unknown")  # Streamlit page doesn't pass this to the crew

        # Validate against Pydantic model (warn, don't crash, to allow graceful display)
        try:
            SkillProfile.model_validate(output)
        except Exception as e:
            log.warning("profile_schema_mismatch", error=str(e))
            # Return raw output anyway so Streamlit can display it

        log.info("profile_crew_complete", user_id=self.user_id)
        return output