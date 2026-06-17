import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.langgraph_workflow.state import initial_state
from src.langgraph_workflow.nodes.profile_ingestion_node import profile_ingestion_node
from src.langgraph_workflow.nodes.skill_assessment_node import skill_assessment_node
from src.langgraph_workflow.nodes.learning_path_node import learning_path_node
from src.langgraph_workflow.nodes.project_generation_node import project_generation_node
from src.langgraph_workflow.nodes.llm_fine_tuning_node import llm_fine_tuning_node
from src.langgraph_workflow.nodes.progress_report_node import progress_report_node
from src.langgraph_workflow.nodes.hitl_node import hitl_node
from src.utils.error_handling import CrewExecutionError


@pytest.fixture
def base_state():
    return initial_state(
        user_id="test_user",
        target_role="ML Engineer",
        session_id="test_session",
        github_profile_url="https://github.com/test",
        kaggle_username="test",
        uploaded_document_paths=["resume.pdf"],
    )


@pytest.mark.asyncio
async def test_profile_ingestion_node_success(base_state):
    with patch("src.langgraph_workflow.nodes.profile_ingestion_node._run_crew", new_callable=AsyncMock) as mock:
        mock.return_value = {"skills": [{"name": "python", "level": 3}]}
        result = await profile_ingestion_node(base_state)
        assert "skill_profile" in result
        assert result["error_context"] is None


@pytest.mark.asyncio
async def test_profile_ingestion_node_failure(base_state):
    with patch("src.langgraph_workflow.nodes.profile_ingestion_node._run_crew", side_effect=CrewExecutionError("fail")):
        result = await profile_ingestion_node(base_state)
        assert result["error_context"] is not None
        assert result["error_context"]["node"] == "profile_ingestion"


@pytest.mark.asyncio
async def test_skill_assessment_node_success(base_state):
    state = {**base_state, "skill_profile": {"skills": []}}
    with patch("src.langgraph_workflow.nodes.skill_assessment_node._run_crew", new_callable=AsyncMock) as mock:
        mock.return_value = [{"skill_name": "LangGraph", "current_level": 1, "required_level": 4, "gap_severity": 2, "weeks_to_close": 4, "learning_objective": "learn", "priority_rank": 1}]
        result = await skill_assessment_node(state)
        assert "skill_gaps" in result
        assert len(result["skill_gaps"]) == 1


@pytest.mark.asyncio
async def test_learning_path_node_success(base_state):
    state = {**base_state, "skill_gaps": []}
    sample_path = {"user_id": "test", "target_role": "ML Engineer", "duration_weeks": 4, "hours_per_week": 10, "skill_gaps": [], "weeks": [], "total_hours": 0, "version": 1, "validation_notes": []}
    with patch("src.langgraph_workflow.nodes.learning_path_node._run_crew", new_callable=AsyncMock) as mock:
        mock.return_value = sample_path
        result = await learning_path_node(state)
        assert "learning_path" in result


@pytest.mark.asyncio
async def test_project_generation_node_parallel(base_state):
    state = {**base_state, "skill_gaps": [{"skill_name": "Python", "gap_severity": 3, "current_level": 1, "weeks_to_close": 2}]}
    sample_project = {"title": "Project", "description": "desc", "problem_statement": "prob", "primary_skill": "Python", "requirements": ["req"], "acceptance_criteria": ["ac"], "estimated_hours": 10, "artifact_type": "API", "difficulty": 3, "created_at": "2025-01-01T00:00:00"}
    with patch("src.langgraph_workflow.nodes.project_generation_node._generate_for_gap", new_callable=AsyncMock) as mock:
        mock.return_value = [sample_project]
        result = await project_generation_node(state)
        assert "practice_projects" in result
        assert len(result["practice_projects"]) == 1


@pytest.mark.asyncio
async def test_llm_fine_tuning_node_skipped(base_state):
    state = {**base_state, "session_notes": []}
    result = await llm_fine_tuning_node(state)
    assert result["fine_tuning_status"] == "skipped"


@pytest.mark.asyncio
async def test_progress_report_node_success(base_state):
    state = {**base_state, "current_week": 1, "learning_path": {"weeks": [{"topics": ["Python"], "estimated_hours": 5}]}, "practice_projects": [], "skill_gaps": []}
    with patch("src.langgraph_workflow.nodes.progress_report_node._run_crew", new_callable=AsyncMock) as mock:
        mock.return_value = {"headline_stat": "80%"}
        result = await progress_report_node(state)
        assert "weekly_report" in result


@pytest.mark.asyncio
async def test_hitl_node_interrupt_resume(base_state):
    state = {**base_state, "current_week": 1, "revision_cycle": 0, "weekly_report": {}}
    with patch("src.langgraph_workflow.nodes.hitl_node.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"hitl_action": "approve", "user_feedback": None}
        result = await hitl_node(state)
        assert result["hitl_action"] == "approve"
        assert result["revision_cycle"] == 0

        mock_interrupt.return_value = {"hitl_action": "revise", "user_feedback": "too easy"}
        result = await hitl_node(state)
        assert result["hitl_action"] == "revise"
        assert result["revision_cycle"] == 1