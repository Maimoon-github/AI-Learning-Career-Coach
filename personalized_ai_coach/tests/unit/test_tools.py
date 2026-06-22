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

# ---------------------------------------------------------------------------
# WebSearchTool – comprehensive test suite
# ---------------------------------------------------------------------------

import src.tools.web_search_tool as _ws_module


@pytest.fixture(autouse=True)
def clear_search_cache():
    """Wipe the module-level TTL cache before every test for isolation."""
    _ws_module._CACHE.clear()
    yield
    _ws_module._CACHE.clear()


_DDG_RAW = [
    {"title": "Result 1", "href": "http://example.com/1", "body": "First snippet"},
    {"title": "Result 2", "href": "http://example.com/2", "body": "Second snippet"},
]


def _make_ddgs_mock(raw=_DDG_RAW):
    """Return a configured mock that mimics the DDGS context-manager API."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text.return_value = raw
    return mock_ddgs_instance


# 1. DDG happy path —————————————————————————————————————————————————————————
def test_web_search_ddg_happy_path():
    tool = WebSearchTool()
    with patch("src.tools.web_search_tool.os.getenv", return_value=""), \
         patch("duckduckgo_search.DDGS", return_value=_make_ddgs_mock()):
        result = tool._run("python tutorials", max_results=2)
    assert isinstance(result, str)
    assert "python tutorials" in result.lower()
    assert "Result 1" in result
    assert "http://example.com/1" in result


# 2. SearXNG as fallback provider ———————————————————————————————————————
def test_web_search_searxng_fallback():
    """When DDG fails, SearXNG is called and its results are returned."""
    tool = WebSearchTool()
    ddgs_mock = _make_ddgs_mock()
    ddgs_mock.text.side_effect = ConnectionError("DDG rate-limited")

    searxng_payload = {
        "results": [
            {"title": "SearXNG Result", "url": "http://searxng.com/1", "content": "SearXNG snippet"},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = searxng_payload
    mock_resp.raise_for_status = MagicMock()

    with patch("duckduckgo_search.DDGS", return_value=ddgs_mock), \
         patch("src.tools.web_search_tool.httpx.get", return_value=mock_resp):
        result = tool._run("machine learning jobs", max_results=1)

    assert "SearXNG Result" in result
    assert "http://searxng.com/1" in result


# 3. SearXNG env-var override ————————————————————————————————————————————
def test_web_search_searxng_instance_url_override():
    """SEARXNG_INSTANCE_URL env var is forwarded to the HTTP call."""
    tool = WebSearchTool()
    ddgs_mock = _make_ddgs_mock()
    ddgs_mock.text.side_effect = RuntimeError("DDG down")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": [{"title": "Custom", "url": "http://my-searx.local/r", "content": "ok"}]}
    mock_resp.raise_for_status = MagicMock()

    with patch("duckduckgo_search.DDGS", return_value=ddgs_mock), \
         patch("src.tools.web_search_tool.httpx.get", return_value=mock_resp) as mock_get, \
         patch.dict("os.environ", {"SEARXNG_INSTANCE_URL": "http://my-searx.local"}):
        tool._run("override test", max_results=1)

    call_url = mock_get.call_args[0][0]
    assert call_url.startswith("http://my-searx.local")


# 4. Deduplication ————————————————————————————————————————————————————————
def test_web_search_deduplication():
    duplicate_raw = [
        {"title": "Dup A", "href": "http://dup.com/page", "body": "First"},
        {"title": "Dup B", "href": "http://dup.com/page", "body": "Second (same URL)"},
        {"title": "Unique", "href": "http://unique.com/page", "body": "Unique result"},
    ]
    tool = WebSearchTool()
    with patch("src.tools.web_search_tool.os.getenv", return_value=""), \
         patch("duckduckgo_search.DDGS", return_value=_make_ddgs_mock(duplicate_raw)):
        result = tool._run("dup query", max_results=5)

    # "Dup B" / second occurrence of dup.com must be absent
    assert result.count("http://dup.com/page") == 1
    assert "Unique" in result


# 5. Query sanitisation ———————————————————————————————————————————————————
def test_web_search_query_sanitisation():
    tool = WebSearchTool()
    captured: list[str] = []

    def fake_ddg_call(query, max_results):
        captured.append(query)
        return []

    ddgs_mock = _make_ddgs_mock([])
    ddgs_mock.text.side_effect = fake_ddg_call

    with patch("src.tools.web_search_tool.os.getenv", return_value=""), \
         patch("duckduckgo_search.DDGS", return_value=ddgs_mock):
        tool._run("  python   jobs   ", max_results=3)

    assert captured == ["python jobs"]


# 6. max_results respected ————————————————————————————————————————————————
def test_web_search_max_results_cap():
    tool = WebSearchTool()
    ddgs_mock = _make_ddgs_mock()

    with patch("src.tools.web_search_tool.os.getenv", return_value=""), \
         patch("duckduckgo_search.DDGS", return_value=ddgs_mock):
        tool._run("ai news", max_results=1)

    ddgs_mock.text.assert_called_once_with("ai news", max_results=1)


# 7. All providers fail ———————————————————————————————————————————————————
def test_web_search_all_providers_fail():
    tool = WebSearchTool()
    ddgs_mock = _make_ddgs_mock()
    ddgs_mock.text.side_effect = ConnectionError("DDG offline")

    with patch("src.tools.web_search_tool.os.getenv", return_value=""), \
         patch("duckduckgo_search.DDGS", return_value=ddgs_mock):
        result = tool._run("edge case", max_results=2)

    assert isinstance(result, str)
    assert "failed" in result.lower() or "unavailable" in result.lower()


# 8. DDG partial / missing keys ——————————————————————————————————————————
def test_web_search_ddg_missing_keys():
    partial_raw = [
        {},                                        # completely empty
        {"href": "http://partial.com"},            # missing title & body
        {"title": "Full", "href": "http://full.com", "body": "Full snippet"},
    ]
    tool = WebSearchTool()
    with patch("src.tools.web_search_tool.os.getenv", return_value=""), \
         patch("duckduckgo_search.DDGS", return_value=_make_ddgs_mock(partial_raw)):
        result = tool._run("partial results", max_results=5)

    # Must not raise; known good entry must appear
    assert "Full" in result


# 9. Caching —————————————————————————————————————————————————————————————
def test_web_search_caching():
    tool = WebSearchTool()
    ddgs_mock = _make_ddgs_mock()

    with patch("src.tools.web_search_tool.os.getenv", return_value=""), \
         patch("duckduckgo_search.DDGS", return_value=ddgs_mock):
        result_first = tool._run("cached query", max_results=2)
        result_second = tool._run("cached query", max_results=2)

    # Provider called only once despite two _run invocations
    assert ddgs_mock.text.call_count == 1
    assert result_first == result_second


# 10. Format results – empty list ————————————————————————————————————————
def test_format_results_empty():
    output = WebSearchTool._format_results([], "empty query")
    assert "No results found" in output

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