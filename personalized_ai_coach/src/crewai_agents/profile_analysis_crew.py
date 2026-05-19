from __future__ import annotations

import json
from typing import Any

import structlog
import yaml
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent

from src.tools.github_tool import GitHubTool
from src.tools.tools import DocumentParserTool, KaggleTool
from src.utils.llm_client import get_llm

log = structlog.get_logger(__name__)


def _load_config(section: str) -> tuple[dict, dict]:
    with open("config/agents.yaml") as f:
        agents_cfg = yaml.safe_load(f)[section]
    with open("config/tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)[section]
    return agents_cfg, tasks_cfg


class ProfileAnalysisCrew:
    """
    Orchestrates GitHub Analyst, Kaggle Analyst, Document Processor, and
    Profile Synthesizer to produce a unified SkillProfile JSON.
    """

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

        agents_cfg, self.tasks_cfg = _load_config("profile_analysis")
        llm = get_llm("structured_extraction")

        github_tool = GitHubTool()
        kaggle_tool = KaggleTool()
        doc_tool = DocumentParserTool()

        self.github_analyst = Agent(
            role=agents_cfg["github_analyst"]["role"],
            goal=agents_cfg["github_analyst"]["goal"],
            backstory=agents_cfg["github_analyst"]["backstory"],
            llm=llm,
            tools=[github_tool],
            verbose=agents_cfg["github_analyst"]["verbose"],
            allow_delegation=agents_cfg["github_analyst"]["allow_delegation"],
            max_iter=3,
        )

        self.kaggle_analyst = Agent(
            role=agents_cfg["kaggle_analyst"]["role"],
            goal=agents_cfg["kaggle_analyst"]["goal"],
            backstory=agents_cfg["kaggle_analyst"]["backstory"],
            llm=llm,
            tools=[kaggle_tool],
            verbose=agents_cfg["kaggle_analyst"]["verbose"],
            allow_delegation=agents_cfg["kaggle_analyst"]["allow_delegation"],
            max_iter=3,
        )

        self.document_processor = Agent(
            role=agents_cfg["document_processor"]["role"],
            goal=agents_cfg["document_processor"]["goal"],
            backstory=agents_cfg["document_processor"]["backstory"],
            llm=llm,
            tools=[doc_tool],
            verbose=agents_cfg["document_processor"]["verbose"],
            allow_delegation=agents_cfg["document_processor"]["allow_delegation"],
            max_iter=3,
        )

        self.profile_synthesizer = Agent(
            role=agents_cfg["profile_synthesizer"]["role"],
            goal=agents_cfg["profile_synthesizer"]["goal"],
            backstory=agents_cfg["profile_synthesizer"]["backstory"],
            llm=get_llm("structured_extraction"),
            tools=[],
            verbose=agents_cfg["profile_synthesizer"]["verbose"],
            allow_delegation=agents_cfg["profile_synthesizer"]["allow_delegation"],
            max_iter=5,
        )

    def _build_tasks(self) -> list[Task]:
        tasks = []
        task_agents = []

        if self.github_url:
            t = Task(
                description=self.tasks_cfg["github_analysis"]["description"].format(
                    github_url=self.github_url, user_id=self.user_id
                ),
                expected_output=self.tasks_cfg["github_analysis"]["expected_output"],
                agent=self.github_analyst,
            )
            tasks.append(t)
            task_agents.append(self.github_analyst)

        if self.kaggle_username:
            t = Task(
                description=self.tasks_cfg["kaggle_analysis"]["description"].format(
                    kaggle_url=f"https://kaggle.com/{self.kaggle_username}",
                    user_id=self.user_id,
                ),
                expected_output=self.tasks_cfg["kaggle_analysis"]["expected_output"],
                agent=self.kaggle_analyst,
            )
            tasks.append(t)
            task_agents.append(self.kaggle_analyst)

        if self.document_paths:
            t = Task(
                description=self.tasks_cfg["document_processing"]["description"].format(
                    user_id=self.user_id,
                    document_paths=", ".join(self.document_paths),
                ),
                expected_output=self.tasks_cfg["document_processing"]["expected_output"],
                agent=self.document_processor,
            )
            tasks.append(t)
            task_agents.append(self.document_processor)

        # Synthesis always runs, uses context from above tasks
        synthesis_task = Task(
            description=self.tasks_cfg["profile_synthesis"]["description"],
            expected_output=self.tasks_cfg["profile_synthesis"]["expected_output"],
            agent=self.profile_synthesizer,
            context=tasks,  # Receives all prior task outputs
        )
        tasks.append(synthesis_task)
        task_agents.append(self.profile_synthesizer)

        return tasks

    def kickoff(self) -> dict[str, Any]:
        tasks = self._build_tasks()
        agents: list[BaseAgent] = [
            self.github_analyst, self.kaggle_analyst,
            self.document_processor, self.profile_synthesizer,
        ]
        # Only include agents that have tasks
        active_agents = list({t.agent for t in tasks if t.agent})

        crew = Crew(
            agents=active_agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            memory=True,
        )

        log.info("profile_crew_starting", user_id=self.user_id)
        result = crew.kickoff(inputs={"user_id": self.user_id})

        raw = result.raw if hasattr(result, "raw") else str(result)
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            parsed = {"raw_output": raw}

        log.info("profile_crew_complete", user_id=self.user_id)
        return parsed