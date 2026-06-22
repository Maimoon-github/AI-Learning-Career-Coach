"""
tests/unit/test_tools.py
========================
Unit tests for all tools.  The WebSearchTool suite covers every code path
specified in the refactoring requirements:
  - DDG happy path
  - SearXNG fallback (DDG failure)
  - SearXNG fallback (DDG empty results)
  - Timeout / HTTP error scenarios
  - Retry behavior
  - Deduplication
  - Schema normalisation / missing fields
  - Cache hit/miss
  - Query sanitisation
  - max_results forwarding
  - All-providers-fail graceful degradation
  - Empty list formatting
  - SEARXNG_INSTANCE_URL env-var override
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from src.tools.github_tool import GitHubTool
from src.tools.kaggle_tool import KaggleTool
from src.tools.document_parser_tool import DocumentParserTool
from src.tools.ollama_tool import OllamaTool

import src.tools.web_search_tool as _ws_module
from src.tools.web_search_tool import (
    SearchResult,
    WebSearchTool,
    _DuckDuckGoProvider,
    _SearXNGProvider,
    _deduplicate,
    _format_results,
    _sanitise_query,
)


# ===========================================================================
# Non-web-search tool tests (unchanged)
# ===========================================================================

@pytest.fixture
def mock_github_response():
    return {
        "languages": {"Python": 85.0, "JavaScript": 15.0},
        "frameworks": ["django", "react"],
        "contribution_streak_days": 7,
        "project_complexity_score": 6.5,
        "key_projects": [{"name": "test_repo", "description": "test", "tech_stack": [], "stars": 10, "size_kb": 500}],
        "collaboration_signals": {"public_repos": 5, "followers": 10, "following": 2},
        "raw_url": "https://github.com/testuser",
    }

@pytest.fixture
def mock_kaggle_response():
    return {
        "tier": "Expert",
        "medals": {"gold": 0, "silver": 1, "bronze": 2},
        "ml_domains": ["nlp", "tabular"],
        "notebook_quality_score": 7.5,
        "active_last_year": True,
        "strongest_domain": "nlp",
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
    test_file = tmp_path / "test.txt"
    test_file.write_text("Sample content\nSecond line")
    tool = DocumentParserTool()
    result = tool._run(str(test_file))
    assert "text" in result
    assert "Sample content" in result["text"]


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
        result = tool._run("fine_tuning_dry_run", user_notes=["note1", "note2"])
        assert result["status"] == "dry_run_success"
        assert result["sample_notes_used"] == 2
        result = tool._run("list_models")
        assert "models" in result


@pytest.mark.asyncio
async def test_github_tool_error():
    with patch("github.Github") as MockGithub:
        MockGithub.side_effect = Exception("API error")
        tool = GitHubTool()
        result = tool._run("https://github.com/invalid")
        assert "error" in result


# ===========================================================================
# WebSearchTool — Shared fixtures & helpers
# ===========================================================================

@pytest.fixture(autouse=True)
def clear_search_cache():
    """Isolate every test: wipe the module-level TTL cache before and after."""
    _ws_module._CACHE.clear()
    yield
    _ws_module._CACHE.clear()


def _sr(title="Title", url="http://example.com", snippet="Snippet") -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet)


def _make_ddg_provider(results: list[SearchResult] | Exception) -> _DuckDuckGoProvider:
    """Return a _DuckDuckGoProvider whose search() is stubbed."""
    provider = MagicMock(spec=_DuckDuckGoProvider)
    if isinstance(results, Exception):
        provider.search.side_effect = results
    else:
        provider.search.return_value = results
    return provider


def _make_searxng_provider(results: list[SearchResult] | Exception) -> _SearXNGProvider:
    """Return a _SearXNGProvider whose search() is stubbed."""
    provider = MagicMock(spec=_SearXNGProvider)
    if isinstance(results, Exception):
        provider.search.side_effect = results
    else:
        provider.search.return_value = results
    return provider


def _tool(ddg=None, searxng=None) -> WebSearchTool:
    """Construct a WebSearchTool with injected provider stubs."""
    t = WebSearchTool()
    if ddg is not None:
        t._ddg = ddg
    if searxng is not None:
        t._searxng = searxng
    return t


# ===========================================================================
# 1. DuckDuckGo happy path
# ===========================================================================

def test_ddg_happy_path_returns_str():
    results = [_sr("Python Jobs", "http://ex.com/1", "Great roles"), _sr("ML Courses", "http://ex.com/2", "Learn ML")]
    tool = _tool(ddg=_make_ddg_provider(results))
    output = tool._run("python jobs", max_results=2)

    assert isinstance(output, str)
    assert "python jobs" in output.lower()
    assert "Python Jobs" in output
    assert "http://ex.com/1" in output
    assert "Great roles" in output


def test_ddg_max_results_forwarded():
    ddg = _make_ddg_provider([_sr()])
    tool = _tool(ddg=ddg)
    tool._run("ai jobs", max_results=3)
    ddg.search.assert_called_once_with("ai jobs", 3)


# ===========================================================================
# 2. SearXNG fallback — DDG raises an exception
# ===========================================================================

def test_searxng_fallback_on_ddg_exception():
    ddg = _make_ddg_provider(ConnectionError("DDG rate-limited"))
    sx = _make_searxng_provider([_sr("SearXNG Hit", "http://searxng.com/1", "From searxng")])
    tool = _tool(ddg=ddg, searxng=sx)

    output = tool._run("machine learning", max_results=1)

    assert "SearXNG Hit" in output
    assert "http://searxng.com/1" in output


# ===========================================================================
# 3. SearXNG fallback — DDG returns empty results
# ===========================================================================

def test_searxng_fallback_on_ddg_empty():
    ddg = _make_ddg_provider([])  # empty list → triggers fallback
    sx = _make_searxng_provider([_sr("Fallback Result", "http://fb.com/1", "Fell back")])
    tool = _tool(ddg=ddg, searxng=sx)

    output = tool._run("obscure topic", max_results=2)

    assert "Fallback Result" in output
    sx.search.assert_called_once()


# ===========================================================================
# 4. All providers fail — graceful degradation
# ===========================================================================

def test_all_providers_fail_returns_error_string():
    ddg = _make_ddg_provider(RuntimeError("DDG down"))
    sx = _make_searxng_provider(RuntimeError("SearXNG down"))
    tool = _tool(ddg=ddg, searxng=sx)

    output = tool._run("edge case", max_results=2)

    assert isinstance(output, str)
    assert "failed" in output.lower() or "unavailable" in output.lower()


# ===========================================================================
# 5. Timeout scenario (SearXNG HTTP timeout)
# ===========================================================================

def test_searxng_http_timeout_falls_through_to_error():
    import httpx as _httpx
    ddg = _make_ddg_provider([])   # empty → tries SearXNG
    sx = _make_searxng_provider(_httpx.TimeoutException("Timeout"))
    tool = _tool(ddg=ddg, searxng=sx)

    output = tool._run("timeout test", max_results=2)

    assert isinstance(output, str)
    # Either no results or an explicit error message
    assert "No results found" in output or "failed" in output.lower() or "unavailable" in output.lower()


# ===========================================================================
# 6. Retry behaviour — real tenacity wiring on _DuckDuckGoProvider
# ===========================================================================

def test_ddg_provider_retries_on_transient_error():
    """
    The real _DuckDuckGoProvider must retry up to 3 attempts.
    We patch DDGS so each instantiation raises on the first call then succeeds.
    """
    call_count = 0

    def flaky_text(query, max_results):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("transient")
        return [{"title": "OK", "href": "http://ok.com", "body": "snippet"}]

    ddgs_instance = MagicMock()
    ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
    ddgs_instance.__exit__ = MagicMock(return_value=False)
    ddgs_instance.text.side_effect = flaky_text

    provider = _DuckDuckGoProvider()
    with patch("duckduckgo_search.DDGS", return_value=ddgs_instance):
        results = provider.search("retry test", max_results=1)

    assert len(results) == 1
    assert results[0].title == "OK"
    assert call_count == 3   # failed twice, succeeded on 3rd


# ===========================================================================
# 7. Deduplication correctness
# ===========================================================================

def test_deduplication_removes_duplicate_urls():
    results = [
        _sr("A", "http://dup.com/page", "first"),
        _sr("B", "http://dup.com/page", "second — same URL"),
        _sr("C", "http://unique.com/page", "unique"),
    ]
    unique = _deduplicate(results)
    urls = [r.url for r in unique]

    assert urls.count("http://dup.com/page") == 1
    assert "http://unique.com/page" in urls


def test_deduplication_preserves_order():
    results = [_sr(f"R{i}", f"http://ex.com/{i}", "") for i in range(5)]
    assert _deduplicate(results) == results


def test_deduplication_keeps_no_url_results():
    results = [_sr("No URL", "", "some snippet")]
    assert _deduplicate(results) == results


# ===========================================================================
# 8. Schema normalisation — SearchResult.from_raw
# ===========================================================================

def test_schema_normalisation_full_ddg():
    raw = {"title": "DDG Title", "href": "http://ddg.com", "body": "DDG body"}
    r = SearchResult.from_raw(raw, url_key="href")
    assert r.title == "DDG Title"
    assert r.url == "http://ddg.com"
    assert r.snippet == "DDG body"


def test_schema_normalisation_full_searxng():
    raw = {"title": "SX Title", "url": "http://sx.com", "content": "SX content"}
    r = SearchResult.from_raw(raw, url_key="url")
    assert r.title == "SX Title"
    assert r.url == "http://sx.com"
    assert r.snippet == "SX content"


def test_schema_normalisation_missing_fields():
    r = SearchResult.from_raw({})
    assert r.title == "Untitled"
    assert r.url == ""
    assert r.snippet == ""


def test_schema_normalisation_none_values():
    raw = {"title": None, "href": None, "body": None}
    r = SearchResult.from_raw(raw, url_key="href")
    assert r.title == "Untitled"
    assert r.url == ""
    assert r.snippet == ""


def test_schema_normalisation_snippet_fallback_priority():
    """snippet field checked first, then body, then content."""
    raw = {"title": "T", "url": "http://x.com", "body": "body text", "content": "content text"}
    r = SearchResult.from_raw(raw, url_key="url")
    assert r.snippet == "body text"  # body wins over content


# ===========================================================================
# 9. Cache hit / miss behaviour
# ===========================================================================

def test_cache_miss_calls_provider():
    ddg = _make_ddg_provider([_sr()])
    tool = _tool(ddg=ddg)
    tool._run("unique query", max_results=2)
    ddg.search.assert_called_once()


def test_cache_hit_does_not_call_provider():
    ddg = _make_ddg_provider([_sr()])
    tool = _tool(ddg=ddg)

    r1 = tool._run("cached query", max_results=2)
    r2 = tool._run("cached query", max_results=2)

    assert ddg.search.call_count == 1    # provider called once only
    assert r1 == r2                      # identical output


def test_cache_miss_on_different_max_results():
    ddg = _make_ddg_provider([_sr()])
    tool = _tool(ddg=ddg)

    tool._run("same query", max_results=2)
    tool._run("same query", max_results=5)

    assert ddg.search.call_count == 2    # different key → two misses


# ===========================================================================
# 10. Query sanitisation
# ===========================================================================

def test_sanitise_strips_surrounding_whitespace():
    assert _sanitise_query("  hello  ") == "hello"


def test_sanitise_collapses_internal_whitespace():
    assert _sanitise_query("python   job   search") == "python job search"


def test_sanitise_query_passed_to_provider():
    ddg = _make_ddg_provider([_sr()])
    tool = _tool(ddg=ddg)
    tool._run("  python   jobs   ", max_results=1)
    ddg.search.assert_called_once_with("python jobs", 1)


# ===========================================================================
# 11. max_results boundary values
# ===========================================================================

def test_max_results_minimum():
    ddg = _make_ddg_provider([_sr()])
    tool = _tool(ddg=ddg)
    tool._run("query", max_results=1)
    ddg.search.assert_called_once_with("query", 1)


def test_max_results_maximum():
    ddg = _make_ddg_provider([_sr()])
    tool = _tool(ddg=ddg)
    tool._run("query", max_results=10)
    ddg.search.assert_called_once_with("query", 10)


# ===========================================================================
# 12. Output formatting
# ===========================================================================

def test_format_results_empty_list():
    output = _format_results([], "foo query")
    assert "No results found" in output
    assert "foo query" in output


def test_format_results_header_present():
    output = _format_results([_sr("T", "http://x.com", "S")], "my query")
    assert 'Web Search Results for: "my query"' in output


def test_format_results_snippet_truncated():
    long_snippet = "x" * 300
    results = [_sr("T", "http://x.com", long_snippet)]
    output = _format_results(results, "q")
    # Truncated snippet must end with "..."
    first_snippet_line = [l for l in output.splitlines() if l.strip().startswith("x")][0]
    assert first_snippet_line.strip().endswith("...")


def test_format_results_missing_url_omits_url_line():
    r = SearchResult(title="No URL", url="", snippet="Some snippet")
    output = _format_results([r], "q")
    assert "URL:" not in output
    assert "Some snippet" in output


# ===========================================================================
# 13. SearXNG SEARXNG_INSTANCE_URL env-var override
# ===========================================================================

def test_searxng_uses_custom_instance_url(monkeypatch):
    monkeypatch.setenv("SEARXNG_INSTANCE_URL", "http://my-searx.local")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [{"title": "Custom", "url": "http://my-searx.local/r", "content": "ok"}]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200

    with patch("src.tools.web_search_tool.httpx.get", return_value=mock_resp) as mock_get:
        provider = _SearXNGProvider()
        provider.search("test query", max_results=1)

    called_url = mock_get.call_args[0][0]
    assert called_url == "http://my-searx.local/search"


def test_searxng_default_instance_url_used_when_env_not_set(monkeypatch):
    monkeypatch.delenv("SEARXNG_INSTANCE_URL", raising=False)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.tools.web_search_tool.httpx.get", return_value=mock_resp) as mock_get:
        provider = _SearXNGProvider()
        provider.search("q", max_results=1)

    called_url = mock_get.call_args[0][0]
    assert "searx.be" in called_url