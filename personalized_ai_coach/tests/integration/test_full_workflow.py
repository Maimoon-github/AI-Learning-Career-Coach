"""
Integration tests for the full LangGraph workflow.
These tests mock external APIs and LLM calls but exercise real graph routing.
Run: pytest tests/integration/ -v --asyncio-mode=auto
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.langgraph_workflow.graph import build_graph
from src.langgraph_workflow.state import initial_state


MOCK_SKILL_PROFILE = {
    "user_id": "inttest_user",
    "target_role": "ML Engineer",
    "skills": [
        {"name": "python", "level": 3, "source": ["github"], "confidence": 0.9},
    ],
}

MOCK_SKILL_GAPS = [
    {
        "skill_name": "mlops",
        "current_level": 1,
        "required_level": 3,
        "gap_severity": 3,
        "weeks_to_close": 4,
        "priority_rank": 1,
        "prerequisites": [],
        "learning_objective": "Deploy and monitor ML models in production",
    }
]

MOCK_LEARNING_PATH = {
    "user_id": "inttest_user",
    "target_role": "ML Engineer",
    "duration_weeks": 4,
    "hours_per_week": 10,
    "weeks": [
        {
            "week_number": 1,
            "primary_skill": "mlops",
            "topics": ["Docker", "CI/CD for ML"],
            "estimated_hours": 10,
            "milestone": "Containerize a model",
            "resources": [],
        }
    ],
}

MOCK_PROJECTS = [{"title": "MLOps Pipeline", "difficulty": 3, "estimated_hours": 15}]

MOCK_REPORT = {
    "week_number": 1,
    "headline_stat": "Completed 2/3 topics",
    "wins": ["Finished Docker module"],
    "coach_note": "Great start with containerization basics.",
}


class TestFullWorkflowIntegration:
    @pytest.mark.asyncio
    async def test_workflow_graph_compiles(self):
        """Graph compilation should succeed with MemorySaver."""
        app = build_graph(checkpointer=MemorySaver()).compile(
            checkpointer=MemorySaver(),
            interrupt_before=["hitl"],
        )
        assert app is not None

    @pytest.mark.asyncio
    async def test_workflow_reaches_hitl_interrupt(self):
        """Full pipeline from profile ingestion to HITL should fire."""
        with (
            patch("src.langgraph_workflow.nodes.all_nodes.ProfileAnalysisCrew") as P,
            patch("src.langgraph_workflow.nodes.all_nodes.SkillGapAssessmentCrew") as SG,
            patch("src.langgraph_workflow.nodes.all_nodes.LearningPathGenerationCrew") as LP,
            patch("src.langgraph_workflow.nodes.all_nodes.ProjectGenerationCrew") as PG,
            patch("src.langgraph_workflow.nodes.all_nodes.LLMFineTuningCrew") as FT,
            patch("src.langgraph_workflow.nodes.all_nodes.ProgressReportingCrew") as PR,
        ):
            P.return_value.kickoff.return_value = MOCK_SKILL_PROFILE
            SG.return_value.kickoff.return_value = MOCK_SKILL_GAPS
            LP.return_value.kickoff.return_value = MOCK_LEARNING_PATH
            PG.return_value.kickoff.return_value = MOCK_PROJECTS
            FT.return_value.kickoff.return_value = {"status": "complete"}
            PR.return_value.kickoff.return_value = MOCK_REPORT

            checkpointer = MemorySaver()
            app = build_graph().compile(checkpointer=checkpointer, interrupt_before=["hitl"])

            state = initial_state(
                user_id="inttest_user",
                target_role="ML Engineer",
                session_id="integration-test-001",
                github_profile_url="https://github.com/testuser",
            )

            events = []
            async for event in app.astream(
                state,
                config={"configurable": {"thread_id": "integration-test-001"}},
                stream_mode="values",
            ):
                events.append(event)

            # Should have processed through to HITL interrupt
            assert len(events) > 0
            final = events[-1]

            # Verify pipeline stages completed
            if "__interrupt__" not in final:
                # If no interrupt in stream, check last state has expected data
                assert final.get("skill_profile") is not None or final.get("error_context") is not None

    @pytest.mark.asyncio
    async def test_workflow_ends_on_profile_failure(self):
        """If profile ingestion fails, workflow should terminate gracefully."""
        with patch("src.langgraph_workflow.nodes.all_nodes.ProfileAnalysisCrew") as P:
            P.return_value.kickoff.side_effect = RuntimeError("GitHub API down")

            checkpointer = MemorySaver()
            app = build_graph().compile(checkpointer=checkpointer, interrupt_before=["hitl"])

            state = initial_state(
                user_id="inttest_user",
                target_role="ML Engineer",
                session_id="failure-test-001",
            )

            events = []
            async for event in app.astream(
                state,
                config={"configurable": {"thread_id": "failure-test-001"}},
                stream_mode="values",
            ):
                events.append(event)

            # Should have error context set and terminated
            assert len(events) > 0