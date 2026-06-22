from __future__ import annotations

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
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level TTL cache — shared across all WebSearchTool instances,
# thread-safe thanks to the explicit lock below.
# ---------------------------------------------------------------------------
_CACHE: TTLCache[tuple[str, int], str] = TTLCache(maxsize=256, ttl=3600)
_CACHE_LOCK: Lock = Lock()

# Public SearXNG instance used as DDG fallback.
# Override via env var SEARXNG_INSTANCE_URL for self-hosted deployments.
_DEFAULT_SEARXNG_URL = "https://searx.be"


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of results to return (1–10)",
    )


# ---------------------------------------------------------------------------
# Retry decorator factory
# ---------------------------------------------------------------------------
def _retry_policy() -> Any:
    return retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class WebSearchTool(BaseTool):
    """Search the web using open-source providers (DuckDuckGo → SearXNG fallback)."""

    name: str = "web_search"
    description: str = (
        "Searches the web for current information about job postings, learning resources, "
        "tech trends, and career market data using open-source, no-API-key providers. "
        "Pass a clear, focused search query. Results are cached for 1 hour."
    )
    args_schema: type[BaseModel] = WebSearchInput

    # ------------------------------------------------------------------
    # Public entry point (called by CrewAI)
    # ------------------------------------------------------------------
    def _run(self, query: str, max_results: int = 5) -> str:  # type: ignore[override]
        clean_query = self._sanitise_query(query)
        cache_key = (clean_query, max_results)

        with _CACHE_LOCK:
            if cache_key in _CACHE:
                log.debug("web_search_cache_hit", query=clean_query)
                return _CACHE[cache_key]  # type: ignore[return-value]

        results = self._search_with_fallback(clean_query, max_results)
        formatted = self._format_results(results, clean_query)

        with _CACHE_LOCK:
            _CACHE[cache_key] = formatted

        return formatted

    # ------------------------------------------------------------------
    # Orchestration — DDG first, SearXNG on failure
    # ------------------------------------------------------------------
    def _search_with_fallback(
        self, query: str, max_results: int
    ) -> list[dict[str, str]]:
        # --- Provider 1: DuckDuckGo ---
        try:
            return self._ddg_search(query, max_results)
        except Exception as exc:
            log.warning(
                "ddg_search_failed_falling_back_to_searxng",
                query=query,
                error=str(exc),
            )

        # --- Provider 2: SearXNG ---
        try:
            return self._searxng_search(query, max_results)
        except Exception as exc:
            log.error("all_search_providers_failed", query=query, error=str(exc))
            return [
                {
                    "title": "Search unavailable",
                    "url": "",
                    "snippet": f"All search providers failed: {exc}",
                }
            ]

    # ------------------------------------------------------------------
    # Provider 1 — DuckDuckGo (duckduckgo-search, no API key)
    # ------------------------------------------------------------------
    def _ddg_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        @_retry_policy()
        def _call() -> list[dict[str, str]]:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]

            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=max_results) or []
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in raw
                ]

        return _call()

    # ------------------------------------------------------------------
    # Provider 2 — SearXNG (self-hostable / public instance, no API key)
    # ------------------------------------------------------------------
    def _searxng_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        instance = os.getenv("SEARXNG_INSTANCE_URL", _DEFAULT_SEARXNG_URL).rstrip("/")

        @_retry_policy()
        def _call() -> list[dict[str, str]]:
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
                "language": "en",
            }
            headers = {"User-Agent": "AI-Learning-Career-Coach/1.0 (open-source)"}
            resp = httpx.get(
                f"{instance}/search",
                params=params,
                headers=headers,
                timeout=10.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            raw: list[dict] = resp.json().get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", r.get("snippet", "")),
                }
                for r in raw[:max_results]
            ]

        return _call()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitise_query(query: str) -> str:
        """Strip leading/trailing whitespace and collapse internal whitespace."""
        return re.sub(r"\s+", " ", query.strip())

    @staticmethod
    def _deduplicate(results: list[dict[str, str]]) -> list[dict[str, str]]:
        """Remove results with duplicate URLs; preserve order."""
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for r in results:
            url = r.get("url", "").strip()
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            unique.append(r)
        return unique

    @staticmethod
    def _format_results(results: list[dict[str, str]], query: str) -> str:
        """
        Return a clean, agent-readable string.

        Format:
            Web Search Results for: "<query>"

            1. <title>
               URL: <url>
               <snippet>
            ...
        """
        unique = WebSearchTool._deduplicate(results)

        if not unique:
            return f'Web Search Results for: "{query}"\n\nNo results found.'

        lines: list[str] = [f'Web Search Results for: "{query}"\n']
        for i, r in enumerate(unique, start=1):
            title = (r.get("title", "") or "Untitled").strip() or "Untitled"
            url = r.get("url", "").strip()
            snippet = r.get("snippet", "").strip()

            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   URL: {url}")
            if snippet:
                if len(snippet) > 240:
                    snippet = snippet[:237] + "..."
                lines.append(f"   {snippet}")
            lines.append("")  # blank line between results

        return "\n".join(lines).rstrip()