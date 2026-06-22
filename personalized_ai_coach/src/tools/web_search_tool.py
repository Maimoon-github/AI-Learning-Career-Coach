"""
web_search_tool.py
==================
Open-source, zero-API-key dual-provider search tool for CrewAI agents.

Provider hierarchy
------------------
1. DuckDuckGo  (duckduckgo-search ≥ 6.0.0)  — no key, no registration
2. SearXNG     (self-hosted or public instance via httpx) — no key

Fallback triggers
-----------------
* DuckDuckGo raises any exception
* DuckDuckGo returns an empty result list
* HTTP / timeout errors on either provider

Configuration
-------------
SEARXNG_INSTANCE_URL  — URL of a SearXNG instance (default: https://searx.be)
"""

from __future__ import annotations

import abc
import os
import re
from threading import Lock
from typing import Any

import httpx
import structlog
from cachetools import TTLCache
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_SEARXNG_URL: str = "https://searx.be"
_CACHE_TTL: int = 3600          # seconds
_CACHE_MAXSIZE: int = 256
_HTTP_TIMEOUT: float = 10.0     # seconds for SearXNG requests
_MAX_SNIPPET_LEN: int = 240     # characters before truncation

# ---------------------------------------------------------------------------
# Module-level thread-safe TTL cache (shared across all tool instances)
# ---------------------------------------------------------------------------
_CACHE: TTLCache[tuple[str, int], str] = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)
_CACHE_LOCK: Lock = Lock()


# ---------------------------------------------------------------------------
# Unified result schema
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    """Normalised, provider-agnostic search result."""

    title: str = Field(default="Untitled")
    url: str = Field(default="")
    snippet: str = Field(default="")

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, url_key: str = "url") -> "SearchResult":
        """Build a SearchResult from a raw provider dict, safely."""
        title = str(raw.get("title") or "").strip() or "Untitled"
        url = str(raw.get(url_key) or "").strip()
        snippet = str(
            raw.get("snippet") or raw.get("body") or raw.get("content") or ""
        ).strip()
        return cls(title=title, url=url, snippet=snippet)


# ---------------------------------------------------------------------------
# Retry policy factory
# ---------------------------------------------------------------------------
def _make_retry_policy(
    attempts: int = 3,
    min_wait: float = 0.5,
    max_wait: float = 8.0,
) -> Any:
    """Return a tenacity retry decorator for transient errors."""
    return retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, min=min_wait, max=max_wait),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Abstract provider base
# ---------------------------------------------------------------------------
class _SearchProvider(abc.ABC):
    """Minimal interface for a search backend."""

    @abc.abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Execute a search and return normalised results."""
        ...


# ---------------------------------------------------------------------------
# Provider 1 — DuckDuckGo
# ---------------------------------------------------------------------------
class _DuckDuckGoProvider(_SearchProvider):
    """Wraps duckduckgo-search with retry and normalisation."""

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        @_make_retry_policy()
        def _call() -> list[SearchResult]:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]

            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=max_results) or []
                log.debug("ddg_raw_results", count=len(raw), query=query)
                return [
                    SearchResult.from_raw(r, url_key="href")
                    for r in raw
                ]

        return _call()


# ---------------------------------------------------------------------------
# Provider 2 — SearXNG
# ---------------------------------------------------------------------------
class _SearXNGProvider(_SearchProvider):
    """
    Queries a SearXNG JSON API endpoint.

    Instance URL resolved at call-time so tests can override via env var.
    """

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        instance = os.environ.get("SEARXNG_INSTANCE_URL", _DEFAULT_SEARXNG_URL).rstrip("/")

        @_make_retry_policy()
        def _call() -> list[SearchResult]:
            resp = httpx.get(
                f"{instance}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "language": "en",
                },
                headers={"User-Agent": "AI-Learning-Career-Coach/1.0 (open-source)"},
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
            )
            resp.raise_for_status()
            raw: list[dict] = resp.json().get("results", [])
            log.debug("searxng_raw_results", count=len(raw), query=query, instance=instance)
            return [
                SearchResult.from_raw(r, url_key="url")
                for r in raw[:max_results]
            ]

        return _call()


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to run.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return (1–10).",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class WebSearchTool(BaseTool):
    """
    Search the web using open-source, zero-API-key providers.

    Provider chain: DuckDuckGo (primary) → SearXNG (fallback).
    Results are cached for 1 hour and deduplicated by URL.
    """

    name: str = "web_search"
    description: str = (
        "Searches the web for current information about job postings, learning resources, "
        "tech trends, and career market data. Uses open-source providers — no API key required. "
        "Pass a clear, focused query. Results are cached for 1 hour."
    )
    args_schema: type[BaseModel] = WebSearchInput

    # Overridable at construction time for testing / DI
    _ddg: _SearchProvider = _DuckDuckGoProvider()
    _searxng: _SearchProvider = _SearXNGProvider()

    # ------------------------------------------------------------------
    # Public entry point (CrewAI calls this)
    # ------------------------------------------------------------------
    def _run(  # type: ignore[override]
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        clean = _sanitise_query(query)
        cache_key = (clean, max_results)

        with _CACHE_LOCK:
            if cache_key in _CACHE:
                log.debug("web_search_cache_hit", query=clean)
                return _CACHE[cache_key]  # type: ignore[return-value]

        results = self._fetch(clean, max_results)
        output = _format_results(results, clean)

        with _CACHE_LOCK:
            _CACHE[cache_key] = output

        return output

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _fetch(self, query: str, max_results: int) -> list[SearchResult]:
        """
        Try DDG; fall back to SearXNG if DDG raises or returns nothing.
        """
        # --- DuckDuckGo ---
        try:
            ddg_results = self._ddg.search(query, max_results)
            if ddg_results:
                log.info("web_search_provider_used", provider="duckduckgo", query=query, count=len(ddg_results))
                return ddg_results
            log.warning("ddg_empty_results_falling_back", query=query)
        except (RetryError, Exception) as exc:
            log.warning("ddg_failed_falling_back", query=query, error=str(exc))

        # --- SearXNG ---
        try:
            sx_results = self._searxng.search(query, max_results)
            log.info("web_search_provider_used", provider="searxng", query=query, count=len(sx_results))
            return sx_results
        except (RetryError, Exception) as exc:
            log.error("all_providers_failed", query=query, error=str(exc))
            return [
                SearchResult(
                    title="Search unavailable",
                    url="",
                    snippet=f"All search providers failed: {exc}",
                )
            ]

    # ------------------------------------------------------------------
    # Expose helpers as static / class methods for direct test access
    # ------------------------------------------------------------------
    @staticmethod
    def sanitise_query(query: str) -> str:
        return _sanitise_query(query)

    @staticmethod
    def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
        return _deduplicate(results)

    @staticmethod
    def format_results(results: list[SearchResult], query: str) -> str:
        return _format_results(results, query)


# ---------------------------------------------------------------------------
# Module-level pure helpers (stateless, easily unit-testable)
# ---------------------------------------------------------------------------

def _sanitise_query(query: str) -> str:
    """Collapse whitespace and strip surrounding spaces."""
    return re.sub(r"\s+", " ", query.strip())


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """
    Remove results with duplicate URLs, preserving insertion order.
    Results without a URL are always retained.
    """
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        url = r.url.strip()
        if url:
            if url in seen:
                continue
            seen.add(url)
        unique.append(r)
    return unique


def _format_results(results: list[SearchResult], query: str) -> str:
    """
    Render results into an agent-readable numbered list.

    Example output::

        Web Search Results for: "python jobs"

        1. Python Developer at ACME Corp
           URL: https://example.com/job/123
           Full-stack Python role requiring Django and REST expertise.

        2. ...
    """
    unique = _deduplicate(results)

    if not unique:
        return f'Web Search Results for: "{query}"\n\nNo results found.'

    lines: list[str] = [f'Web Search Results for: "{query}"\n']
    for i, r in enumerate(unique, start=1):
        title = r.title or "Untitled"
        snippet = r.snippet
        if len(snippet) > _MAX_SNIPPET_LEN:
            snippet = snippet[: _MAX_SNIPPET_LEN - 3] + "..."

        lines.append(f"{i}. {title}")
        if r.url:
            lines.append(f"   URL: {r.url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines).rstrip()