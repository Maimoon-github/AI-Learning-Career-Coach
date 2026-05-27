import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.tools.github_tool import GitHubTool
from src.tools.kaggle_tool import KaggleTool
from src.tools.document_parser_tool import DocumentParserTool
from src.tools.web_search_tool import WebSearchTool
from src.tools.ollama_tool import OllamaTool


@pytest.fixture
def mock_github_response():
    return {
        "languages": {"Python": 85.0, "JavaScript": 15.0},
        "frameworks": ["django", "react"],
        "contribution_streak_days": 7,
        "project_complexity_score": 6.5,
        "key_projects": [{"name": "test_repo", "description": "test", "tech_stack": [], "stars": 10, "size_kb": 500}],
        "collaboration_signals": {"public_repos": 5, "followers": 10, "following": 2},
        "raw_url": "https://github.com/testuser"
    }

@pytest.fixture
def mock_kaggle_response():
    return {
        "tier": "Expert",
        "medals": {"gold": 0, "silver": 1, "bronze": 2},
        "ml_domains": ["nlp", "tabular"],
        "notebook_quality_score": 7.5,
        "active_last_year": True,
        "strongest_domain": "nlp"
    }

def test_github_tool(mock_github_response):
    with patch("github.Github") as MockGithub:
        mock_user = MagicMock()
        mock_user.public_repos = 5
        mock_user.followers = 10
        mock_user.following = 2
        mock_user.get_repos.return_value = []
        mock_user.get_events.return_value = []
        MockGithub.return_value.get_user.return_value = mock_user
        tool = GitHubTool()
        result = tool._run("https://github.com/testuser")
        assert "languages" in result
        assert isinstance(result["languages"], dict)

def test_kaggle_tool(mock_kaggle_response):
    with patch("src.tools.kaggle_tool.KaggleTool._async_run") as mock_async_run:
        mock_async_run.return_value = mock_kaggle_response
        tool = KaggleTool()
        result = tool._run("testuser")
        assert "tier" in result
        assert result["tier"] in ("Novice", "Contributor", "Expert", "Master", "Grandmaster")

def test_document_parser_tool(tmp_path):
    # Create a test text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Sample content\nSecond line")
    tool = DocumentParserTool()
    result = tool._run(str(test_file))
    assert "text" in result
    assert "Sample content" in result["text"]

def test_web_search_tool():
    with patch("duckduckgo_search.DDGS") as MockDDGS:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__.return_value.text.return_value = [
            {"title": "Result 1", "href": "http://example.com", "body": "Snippet"}
        ]
        MockDDGS.return_value = mock_ddgs
        tool = WebSearchTool()
        results = tool._run("test query", max_results=1)
        assert len(results) >= 1
        assert "title" in results[0]

def test_ollama_tool():
    with patch("httpx.AsyncClient") as MockClient:
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "llama3.2:3b"}, {"name": "llama3.1:70b"}]}
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response
        mock_client.delete.return_value = mock_response
        MockClient.return_value.__aenter__.return_value = mock_client
        tool = OllamaTool()
        # Dry run
        result = tool._run("fine_tuning_dry_run", user_notes=["note1", "note2"])
        assert result["status"] == "dry_run_success"
        assert result["sample_notes_used"] == 2
        # List models
        result = tool._run("list_models")
        assert "models" in result

@pytest.mark.asyncio
async def test_github_tool_error():
    with patch("github.Github") as MockGithub:
        MockGithub.side_effect = Exception("API error")
        tool = GitHubTool()
        result = tool._run("https://github.com/invalid")
        assert "error" in result