"""
Unit tests for LangGraph nodes, CrewAI crews, and tools.
Run: pytest tests/ -v --asyncio-mode=auto
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.langgraph_workflow.state import initial_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> dict:
    return initial_state(
        user_id="test_user_001",
        target_role="ML Engineer",
        session_id="test-session-001",
        github_profile_url="https://github.com/testuser",
        kaggle_username="testuser",
        uploaded_document_paths=[],
    )


@pytest.fixture
def state_with_profile(base_state) -> dict:
    return {
        **base_state,
        "skill_profile": {
            "user_id": "test_user_001",
            "target_role": "ML Engineer",
            "skills": [
                {"name": "python", "level": 3, "source": ["github"], "confidence": 0.9},
                {"name": "pytorch", "level": 2, "source": ["kaggle"], "confidence": 0.7},
            ],
            "github_signals": {"languages": {"Python": 75.0, "JavaScript": 15.0}},
        },
    }


@pytest.fixture
def state_with_gaps(state_with_profile) -> dict:
    return {
        **state_with_profile,
        "skill_gaps": [
            {
                "skill_name": "mlops",
                "current_level": 1,
                "required_level": 3,
                "gap_severity": 3,
                "weeks_to_close": 4,
                "priority_rank": 1,
            },
            {
                "skill_name": "distributed_training",
                "current_level": 0,
                "required_level": 2,
                "gap_severity": 2,
                "weeks_to_close": 3,
                "priority_rank": 2,
            },
        ],
    }


# ── State tests ───────────────────────────────────────────────────────────────

class TestAgentState:
    def test_initial_state_has_all_keys(self, base_state):
        required_keys = [
            "user_id", "target_role", "session_id", "skill_profile",
            "skill_gaps", "learning_path", "practice_projects",
            "current_week", "hitl_action", "messages",
        ]
        for key in required_keys:
            assert key in base_state, f"Missing key: {key}"

    def test_initial_state_defaults(self, base_state):
        assert base_state["skill_profile"] is None
        assert base_state["skill_gaps"] == []
        assert base_state["learning_path"] is None
        assert base_state["current_week"] == 1
        assert base_state["revision_cycle"] == 0
        assert base_state["messages"] == []


# ── Node tests ────────────────────────────────────────────────────────────────

class TestProfileIngestionNode:
    @pytest.mark.asyncio
    async def test_successful_profile_ingestion(self, base_state):
        mock_profile = {
            "user_id": "test_user_001",
            "skills": [{"name": "python", "level": 3, "confidence": 0.9}],
        }
        with patch(
            "src.langgraph_workflow.nodes.all_nodes.ProfileAnalysisCrew"
        ) as MockCrew:
            MockCrew.return_value.kickoff.return_value = mock_profile
            from src.langgraph_workflow.nodes.all_nodes import profile_ingestion_node
            result = await profile_ingestion_node(base_state)

        assert result["skill_profile"] == mock_profile
        assert result["error_context"] is None

    @pytest.mark.asyncio
    async def test_profile_ingestion_handles_error(self, base_state):
        with patch(
            "src.langgraph_workflow.nodes.all_nodes.ProfileAnalysisCrew"
        ) as MockCrew:
            MockCrew.return_value.kickoff.side_effect = RuntimeError("API unavailable")
            from src.langgraph_workflow.nodes.all_nodes import profile_ingestion_node
            result = await profile_ingestion_node(base_state)

        assert result["error_context"] is not None
        assert "profile_ingestion" in result["error_context"]["node"]


class TestSkillAssessmentNode:
    @pytest.mark.asyncio
    async def test_skill_assessment_produces_gaps(self, state_with_profile):
        mock_gaps = [
            {"skill_name": "mlops", "current_level": 1, "required_level": 3, "gap_severity": 3},
        ]
        with patch(
            "src.langgraph_workflow.nodes.all_nodes.SkillGapAssessmentCrew"
        ) as MockCrew:
            MockCrew.return_value.kickoff.return_value = mock_gaps
            from src.langgraph_workflow.nodes.all_nodes import skill_assessment_node
            result = await skill_assessment_node(state_with_profile)

        assert result["skill_gaps"] == mock_gaps
        assert result["error_context"] is None

    @pytest.mark.asyncio
    async def test_skill_assessment_fails_without_profile(self, base_state):
        from src.langgraph_workflow.nodes.all_nodes import skill_assessment_node
        result = await skill_assessment_node(base_state)
        assert result["error_context"] is not None


class TestProjectGenerationNode:
    @pytest.mark.asyncio
    async def test_generates_projects_for_top_gaps_in_parallel(self, state_with_gaps):
        mock_projects = [{"title": "MLOps Pipeline", "difficulty": 3}]
        with patch(
            "src.langgraph_workflow.nodes.all_nodes.ProjectGenerationCrew"
        ) as MockCrew:
            MockCrew.return_value.kickoff.return_value = mock_projects
            from src.langgraph_workflow.nodes.all_nodes import project_generation_node
            result = await project_generation_node(state_with_gaps)

        assert len(result["practice_projects"]) > 0

    @pytest.mark.asyncio
    async def test_no_projects_when_no_gaps(self, state_with_profile):
        from src.langgraph_workflow.nodes.all_nodes import project_generation_node
        result = await project_generation_node(state_with_profile)
        assert result["practice_projects"] == []


# ── Routing logic tests ───────────────────────────────────────────────────────

class TestGraphRouting:
    def test_route_after_profile_goes_to_assessment_on_success(self, state_with_profile):
        from src.langgraph_workflow.graph import route_after_profile
        result = route_after_profile(state_with_profile)
        assert result == "skill_assessment"

    def test_route_after_profile_ends_on_missing_profile(self, base_state):
        from langgraph.graph import END
        from src.langgraph_workflow.graph import route_after_profile
        result = route_after_profile({**base_state, "error_context": {"error": "failed"}})
        assert result == END

    def test_route_after_hitl_approve_advances(self, state_with_gaps):
        state = {**state_with_gaps, "hitl_action": "approve", "revision_cycle": 0}
        from src.langgraph_workflow.graph import route_after_hitl
        result = route_after_hitl(state)
        assert result == "learning_path"

    def test_route_after_hitl_end_terminates(self, state_with_gaps):
        from langgraph.graph import END
        state = {**state_with_gaps, "hitl_action": "end", "revision_cycle": 0}
        from src.langgraph_workflow.graph import route_after_hitl
        result = route_after_hitl(state)
        assert result == END


# ── Model tests ───────────────────────────────────────────────────────────────

class TestSkillProfileModel:
    def test_skill_level_lookup(self):
        from src.models.skill_profile_model import SkillEntry, SkillLevel, SkillProfile
        profile = SkillProfile(
            user_id="u1",
            target_role="ML Engineer",
            skills=[SkillEntry(name="Python", level=SkillLevel.INTERMEDIATE, confidence=0.9)],
        )
        assert profile.get_skill_level("python") == SkillLevel.INTERMEDIATE
        assert profile.get_skill_level("unknown_skill") == SkillLevel.NONE

    def test_skill_name_normalization(self):
        from src.models.skill_profile_model import SkillEntry, SkillLevel
        entry = SkillEntry(name="  Python  ", level=SkillLevel.ADVANCED, confidence=1.0)
        assert entry.name == "python"

    def test_top_skills_ordering(self):
        from src.models.skill_profile_model import SkillEntry, SkillLevel, SkillProfile
        profile = SkillProfile(
            user_id="u1",
            target_role="DE",
            skills=[
                SkillEntry(name="sql", level=SkillLevel.EXPERT, confidence=0.95),
                SkillEntry(name="python", level=SkillLevel.INTERMEDIATE, confidence=0.8),
                SkillEntry(name="spark", level=SkillLevel.ADVANCED, confidence=0.85),
            ],
        )
        top = profile.top_skills(n=2)
        assert top[0].name == "sql"
        assert top[1].name == "spark"


# ── Error handling tests ──────────────────────────────────────────────────────

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_third_attempt(self):
        from src.utils.error_handling import ToolExecutionError, with_retry

        call_count = 0

        @with_retry(max_attempts=3, retriable_errors=(ToolExecutionError,))
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ToolExecutionError("transient failure")
            return "success"

        result = await flaky_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_raises_after_max_attempts(self):
        from src.utils.error_handling import ToolExecutionError, with_retry

        @with_retry(max_attempts=2, retriable_errors=(ToolExecutionError,))
        async def always_fails():
            raise ToolExecutionError("permanent failure")

        with pytest.raises(ToolExecutionError):
            await always_fails()

    def test_non_retriable_error_is_not_retried(self):
        from src.utils.error_handling import ToolExecutionError, with_retry

        call_count = 0

        @with_retry(max_attempts=3, retriable_errors=(ToolExecutionError,))
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retriable")

        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(raises_value_error())

        assert call_count == 1  # No retries for non-retriable errors