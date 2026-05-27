import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.crewai_agents.profile_analysis_crew import ProfileAnalysisCrew
from src.crewai_agents.skill_gap_assessment_crew import SkillGapAssessmentCrew
from src.crewai_agents.learning_path_generation_crew import LearningPathGenerationCrew
from src.crewai_agents.project_generation_crew import ProjectGenerationCrew
from src.crewai_agents.llm_fine_tuning_crew import LLMFineTuningCrew
from src.crewai_agents.progress_reporting_crew import ProgressReportingCrew
from src.utils.error_handling import CrewExecutionError


@pytest.fixture
def mock_llm():
    with patch("src.utils.llm_client.get_llm") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_tools():
    with patch("src.tools.github_tool.GitHubTool") as gh, \
         patch("src.tools.kaggle_tool.KaggleTool") as kg, \
         patch("src.tools.document_parser_tool.DocumentParserTool") as dp, \
         patch("src.tools.web_search_tool.WebSearchTool") as ws, \
         patch("src.tools.ollama_tool.OllamaTool") as ot:
        gh.return_value._run.return_value = {"languages": {"Python": 80}}
        kg.return_value._run.return_value = {"tier": "Expert"}
        dp.return_value._run.return_value = {"text": "Skilled in ML"}
        ws.return_value._run.return_value = [{"title": "Resource", "url": "http://example.com"}]
        ot.return_value._run.return_value = {"status": "dry_run_success"}
        yield


def test_profile_analysis_crew_instantiation(mock_llm, mock_tools):
    crew = ProfileAnalysisCrew("test_user", github_url="https://github.com/test", kaggle_username="test")
    assert crew.github_analyst is not None
    assert crew.kaggle_analyst is not None
    assert crew.doc_processor is not None
    assert crew.synthesizer is not None


def test_skill_gap_assessment_crew_output(mock_llm, mock_tools):
    with patch("crewai.Crew.kickoff") as mock_kickoff:
        mock_kickoff.return_value.raw = json.dumps([
            {"skill_name": "LangGraph", "current_level": 1, "required_level": 4, "gap_severity": 2,
             "weeks_to_close": 4, "prerequisites": [], "learning_objective": "Learn LangGraph", "priority_rank": 1}
        ])
        crew = SkillGapAssessmentCrew({"skills": []}, "ML Engineer")
        gaps = crew.kickoff()
        assert len(gaps) >= 1
        assert gaps[0]["skill_name"] == "LangGraph"


def test_learning_path_generation_crew_handoff(mock_llm, mock_tools):
    sample_path = {
        "user_id": "test", "target_role": "ML Engineer", "duration_weeks": 4, "hours_per_week": 10,
        "skill_gaps": [], "weeks": [{"week_number": 1, "primary_skill": "Python", "topics": ["basics"],
                                    "estimated_hours": 5, "milestone": "Milestone", "is_review_week": False}],
        "total_hours": 20, "version": 1, "validation_notes": []
    }
    with patch("crewai.Crew.kickoff") as mock_kickoff:
        mock_kickoff.return_value.raw = json.dumps(sample_path)
        crew = LearningPathGenerationCrew([{"skill_name": "Python"}], duration_weeks=4)
        result = crew.kickoff()
        assert "weeks" in result
        assert result["duration_weeks"] == 4


def test_project_generation_crew_produces_list(mock_llm, mock_tools):
    sample_projects = [
        {"title": "Project 1", "description": "Detailed description of project 1 with more than ten characters.",
         "problem_statement": "Detailed problem statement for project 1 with more than ten characters.",
         "primary_skill": "Python", "requirements": ["req1"], "acceptance_criteria": ["crit1"],
         "estimated_hours": 10, "artifact_type": "API", "difficulty": 3, "created_at": "2025-01-01T00:00:00"}
    ]
    with patch("crewai.Crew.kickoff") as mock_kickoff:
        mock_kickoff.return_value.raw = json.dumps(sample_projects)
        crew = ProjectGenerationCrew({"skill_name": "Python"}, current_level=2, available_hours=20)
        projects = crew.kickoff()
        assert isinstance(projects, list)
        assert len(projects) >= 1
        assert projects[0]["title"] == "Project 1"


def test_llm_fine_tuning_crew_status(mock_llm, mock_tools):
    with patch("crewai.Crew.kickoff") as mock_kickoff, \
         patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_file = MagicMock()
        mock_file.name = "test_file.json"
        mock_temp.return_value.__enter__.return_value = mock_file
        with patch("os.unlink"):
            mock_kickoff.return_value.raw = json.dumps({"status": "dry_run_success", "sample_notes_used": 10})
            crew = LLMFineTuningCrew("test", ["note1", "note2"], base_model="llama3.2:3b")
            result = crew.kickoff()
            assert result.get("status") == "dry_run_success"


def test_progress_reporting_crew_output(mock_llm, mock_tools):
    sample_report = {"headline_stat": "80% completion", "wins": ["Finished project"], "coach_note": "Good job"}
    with patch("crewai.Crew.kickoff") as mock_kickoff:
        mock_kickoff.return_value.raw = json.dumps(sample_report)
        crew = ProgressReportingCrew("test", {}, 1, {})
        report = crew.kickoff()
        assert "headline_stat" in report


def test_crew_failure_raises_crew_execution_error(mock_llm, mock_tools):
    with patch("crewai.Crew.kickoff", side_effect=Exception("Simulated failure")):
        crew = ProfileAnalysisCrew("test_user")
        with pytest.raises(CrewExecutionError):
            crew.kickoff()