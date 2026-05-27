from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import structlog
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, le=10)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Searches the web for current information. Use for job postings, learning resources, "
        "and market data. Results are cached for 1 hour."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _cached_search(self, query: str, max_results: int) -> tuple[tuple[dict[str, str], ...], str]:
        """Cache results for 1 hour (TTL handled by lru_cache with time-based invalidation)."""
        # Simple TTL: we'll rely on cachetools or manual timestamp; here we use lru_cache without TTL,
        # but we'll add a timestamp in the result and ignore cache after 3600 seconds.
        # For production, use cachetools.TTLCache.
        from cachetools import TTLCache
        cache = getattr(self, "_search_cache", None)
        if cache is None:
            self._search_cache = TTLCache(maxsize=128, ttl=3600)
        key = (query, max_results)
        if key in self._search_cache:
            return self._search_cache[key]
        results = self._perform_search(query, max_results)
        self._search_cache[key] = results
        return results

    def _perform_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        try:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_key)
                response = client.search(query=query, max_results=max_results)
                return [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                    for r in response.get("results", [])
                ]
            else:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    return [
                        {"title": r["title"], "url": r["href"], "snippet": r["body"]}
                        for r in ddgs.text(query, max_results=max_results)
                    ]
        except Exception as exc:
            log.error("web_search_error", query=query, error=str(exc))
            return [{"title": "Search failed", "url": "", "snippet": str(exc)}]

    def _run(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return self._cached_search(query, max_results)