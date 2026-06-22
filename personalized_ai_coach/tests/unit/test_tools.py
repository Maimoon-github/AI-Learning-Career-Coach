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

  GitHubTool suite covers:
  - Happy path with mocked httpx responses
  - Cache hit/miss
  - Invalid / unparseable username
  - Network error graceful degradation
  - Username parsing helper
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

from src.tools.github_tool import GitHubTool, _parse_username, _CACHE
import src.tools.github_tool as _gh_module
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
# Shared fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def clear_github_cache():
    """Isolate every test: wipe the module-level GitHub TTL cache."""
    _gh_module._CACHE.clear()
    yield
    _gh_module._CACHE.clear()


_MOCK_USER = {
    "login": "testuser",
    "name": "Test User",
    "bio": "A test user",
    "public_repos": 5,
    "public_gists": 2,
    "followers": 10,
    "following": 2,
    "hireable": None,
    "company": "",
    "location": "Earth",
    "blog": "",
    "twitter_username": "",
    "created_at": "2020-01-01T00:00:00Z",
}

_MOCK_REPOS = [
    {
        "name": "test_repo",
        "fork": False,
        "stargazers_count": 10,
        "forks_count": 2,
        "size": 500,
        "language": "Python",
        "description": "A test repository",
        "topics": ["django", "react"],
        "open_issues_count": 1,
        "updated_at": "2024-01-01T00:00:00Z",
        "html_url": "https://github.com/testuser/test_repo",
        "default_branch": "main",
        "languages_url": "https://api.github.com/repos/testuser/test_repo/languages",
    }
]

_MOCK_LANGS = {"Python": 85000, "JavaScript": 15000}


def _make_mock_httpx_response(json_data, status_code=200):
    """Build a MagicMock that looks like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    resp.headers = {}
    return resp


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


# ===========================================================================
# GitHubTool — username parsing unit tests
# ===========================================================================

def test_parse_username_from_full_url():
    assert _parse_username("https://github.com/testuser") == "testuser"

def test_parse_username_from_url_with_trailing_slash():
    assert _parse_username("https://github.com/testuser/") == "testuser"

def test_parse_username_bare():
    assert _parse_username("testuser") == "testuser"

def test_parse_username_at_prefix():
    assert _parse_username("@testuser") == "testuser"

def test_parse_username_invalid_raises():
    with pytest.raises(ValueError):
        _parse_username("")

def test_parse_username_invalid_chars_raises():
    with pytest.raises(ValueError):
        _parse_username("user name with spaces")


# ===========================================================================
# GitHubTool — happy path with httpx mock
# ===========================================================================

def test_github_tool_happy_path():
    """
    Mock the entire _async_run coroutine so we don't hit the network.
    Verify the tool's _run bridges correctly and output schema is intact.
    """
    expected = {
        "languages": {"Python": 85.0, "JavaScript": 15.0},
        "frameworks": ["django", "react"],
        "contribution_streak_days": 7,
        "project_complexity_score": 6.5,
        "key_projects": [{"name": "test_repo", "description": "A test repository",
                          "tech_stack": ["django", "react"], "stars": 10, "size_kb": 500,
                          "forks": 2, "language": "Python", "open_issues": 1,
                          "last_updated": "2024-01-01T00:00:00Z",
                          "url": "https://github.com/testuser/test_repo"}],
        "collaboration_signals": {"public_repos": 5, "followers": 10, "following": 2,
                                  "public_gists": 2, "total_stars_earned": 10,
                                  "total_forks_earned": 2, "hireable": None, "company": "",
                                  "location": "Earth", "blog": "", "twitter_username": ""},
        "raw_url": "https://github.com/testuser",
        "username": "testuser",
        "name": "Test User",
        "bio": "A test user",
        "account_created": "2020-01-01T00:00:00Z",
        "activity_days_last_year": 0,
        "total_repos_analyzed": 1,
        "top_language": "Python",
    }

    tool = GitHubTool()
    with patch.object(tool, "_async_run", new=AsyncMock(return_value=expected)):
        result = tool._run("https://github.com/testuser")

    assert "languages" in result
    assert isinstance(result["languages"], dict)
    assert "frameworks" in result
    assert "key_projects" in result
    assert "collaboration_signals" in result
    assert result["username"] == "testuser"
    assert result["top_language"] == "Python"


# ===========================================================================
# GitHubTool — _async_run with httpx mocked at the client level
# ===========================================================================

@pytest.mark.asyncio
async def test_github_tool_async_run_mocked():
    """
    Mock httpx.AsyncClient.get to return preset responses per URL pattern.
    Validates that _analyze_profile assembles output correctly.
    """
    tool = GitHubTool()

    async def fake_get(url, **kwargs):
        if "/users/testuser" in url and "repos" not in url and "events" not in url:
            return _make_mock_httpx_response(_MOCK_USER)
        elif "/repos" in url and "/languages" not in url:
            return _make_mock_httpx_response(_MOCK_REPOS)
        elif "languages" in url:
            return _make_mock_httpx_response(_MOCK_LANGS)
        elif "events" in url:
            return _make_mock_httpx_response([])
        return _make_mock_httpx_response({})

    mock_client = AsyncMock()
    mock_client.get.side_effect = fake_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.tools.github_tool.httpx.AsyncClient", return_value=mock_client):
        result = await tool._async_run("https://github.com/testuser", max_repos=5)

    assert "languages" in result
    assert isinstance(result.get("project_complexity_score"), float)
    assert "collaboration_signals" in result


# ===========================================================================
# GitHubTool — error and edge-case tests
# ===========================================================================

def test_github_tool_invalid_username():
    """Empty URL should return an error dict without raising."""
    tool = GitHubTool()
    result = tool._run("")
    assert "error" in result
    assert result["languages"] == {}


def test_github_tool_network_error():
    """Network failures must be caught and returned as error dict."""
    tool = GitHubTool()
    with patch.object(tool, "_async_run", new=AsyncMock(side_effect=Exception("network down"))):
        # _run wraps _async_run; but since we patch _async_run directly,
        # _run will propagate — so we test _async_run error path separately.
        pass

    # Test the actual error path inside _async_run
    import asyncio
    async def run():
        with patch("src.tools.github_tool._analyze_profile", side_effect=Exception("boom")):
            return await tool._async_run("https://github.com/testuser")

    result = asyncio.run(run())
    assert "error" in result
    assert "languages" in result


def test_github_tool_user_not_found():
    """404 response for user endpoint returns error dict."""
    tool = GitHubTool()

    import asyncio

    async def fake_get(url, **kwargs):
        return _make_mock_httpx_response({}, status_code=404)

    mock_client = AsyncMock()
    mock_client.get.side_effect = fake_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def run():
        with patch("src.tools.github_tool.httpx.AsyncClient", return_value=mock_client):
            return await tool._async_run("https://github.com/nobody_exists_xyz")

    result = asyncio.run(run())
    assert "error" in result


# ===========================================================================
# GitHubTool — caching behaviour
# ===========================================================================

def test_github_tool_cache_hit_does_not_rerun():
    """Second call with same args returns cached result without calling _analyze_profile."""
    tool = GitHubTool()
    first_result = {
        "languages": {"Python": 100.0},
        "frameworks": [],
        "contribution_streak_days": 3,
        "project_complexity_score": 2.0,
        "key_projects": [],
        "collaboration_signals": {},
        "raw_url": "https://github.com/cacheduser",
        "username": "cacheduser",
        "name": "Cached User",
        "bio": "",
        "account_created": "",
        "activity_days_last_year": 0,
        "total_repos_analyzed": 0,
        "top_language": "Python",
    }
    call_count = 0

    async def fake_analyze(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return first_result

    with patch("src.tools.github_tool._analyze_profile", side_effect=fake_analyze):
        r1 = tool._run("https://github.com/cacheduser", max_repos=5)
        r2 = tool._run("https://github.com/cacheduser", max_repos=5)

    assert call_count == 1       # second call served from cache
    assert r1 == r2
    assert r1["top_language"] == "Python"


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