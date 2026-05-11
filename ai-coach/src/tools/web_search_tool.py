"""DuckDuckGo search tool."""

# src/tools/web_search_tool.py

from crewai_tools import tool
from duckduckgo_search import DDGS   # pip install duckduckgo-search


@tool("duckduckgo_search")
def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo (no API key required).
    Returns a list of results with title, url, and snippet.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]