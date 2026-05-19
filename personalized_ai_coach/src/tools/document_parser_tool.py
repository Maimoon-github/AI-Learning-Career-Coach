from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import structlog
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


# ── Kaggle Tool ───────────────────────────────────────────────────────────────

class KaggleInput(BaseModel):
    username: str = Field(description="Kaggle username")


class KaggleTool(BaseTool):
    name: str = "kaggle_profile_analyzer"
    description: str = "Fetches and evaluates a Kaggle profile: competition tier, medals, ML domains, notebook quality."
    args_schema: type[BaseModel] = KaggleInput

    def _run(self, username: str) -> dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(self._async_run(username))

    async def _async_run(self, username: str) -> dict[str, Any]:
        try:
            import kaggle  # noqa: F401
            from kaggle.api.kaggle_api_extended import KaggleApiExtended

            api = KaggleApiExtended()
            api.authenticate()

            # User competitions
            competitions = api.competitions_list(search=username)
            ml_domains: set[str] = set()
            domain_map = {
                "nlp": ["nlp", "text", "language", "sentiment"],
                "computer_vision": ["image", "vision", "detection", "segmentation"],
                "tabular": ["tabular", "regression", "classification", "feature"],
                "time_series": ["time-series", "forecast", "temporal"],
            }

            medals: dict[str, int] = {"gold": 0, "silver": 0, "bronze": 0}
            for comp in competitions:
                title_lower = (getattr(comp, "title", "") or "").lower()
                for domain, keywords in domain_map.items():
                    if any(kw in title_lower for kw in keywords):
                        ml_domains.add(domain)

            tier_map = ["Novice", "Contributor", "Expert", "Master", "Grandmaster"]
            tier_score = min(len(competitions) // 5, 4)

            log.info("kaggle_analysis_complete", username=username)
            return {
                "tier": tier_map[tier_score],
                "medals": medals,
                "ml_domains": sorted(ml_domains),
                "notebook_quality_score": min(10, tier_score * 2.5),
                "active_last_year": len(competitions) > 0,
                "strongest_domain": sorted(ml_domains)[0] if ml_domains else "tabular",
            }
        except Exception as exc:
            log.error("kaggle_tool_error", error=str(exc))
            return {
                "tier": "Novice", "medals": {}, "ml_domains": [],
                "notebook_quality_score": 0, "active_last_year": False,
                "strongest_domain": "", "error": str(exc),
            }


# ── Web Search Tool ───────────────────────────────────────────────────────────

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, le=10)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Searches the web for current information. Use for job postings, learning resources, and market data."
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        try:
            # Prefer Tavily for higher quality; fallback to DuckDuckGo
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


# ── Document Parser Tool ──────────────────────────────────────────────────────

class DocumentParserInput(BaseModel):
    file_path: str = Field(description="Absolute path to a PDF, Markdown, or text file")


class DocumentParserTool(BaseTool):
    name: str = "document_parser"
    description: str = "Parses PDF, Markdown, and text documents to extract structured text content."
    args_schema: type[BaseModel] = DocumentParserInput

    def _run(self, file_path: str) -> dict[str, str]:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}", "text": ""}

        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                return self._parse_pdf(path)
            elif suffix in (".md", ".markdown"):
                return {"text": path.read_text(encoding="utf-8"), "format": "markdown"}
            elif suffix in (".txt", ".rst"):
                return {"text": path.read_text(encoding="utf-8"), "format": "text"}
            elif suffix in (".docx",):
                return self._parse_docx(path)
            else:
                return {"error": f"Unsupported format: {suffix}", "text": ""}
        except Exception as exc:
            log.error("document_parser_error", path=str(path), error=str(exc))
            return {"error": str(exc), "text": ""}

    def _parse_pdf(self, path: Path) -> dict[str, str]:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return {"text": text, "format": "pdf", "pages": len(reader.pages)}

    def _parse_docx(self, path: Path) -> dict[str, str]:
        from docx import Document
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return {"text": text, "format": "docx"}


# ── Ollama Tool ───────────────────────────────────────────────────────────────

class OllamaToolInput(BaseModel):
    action: str = Field(description="Action: list_models | pull_model | delete_model | model_info")
    model_name: str = Field(default="", description="Model name for pull/delete/info actions")


class OllamaTool(BaseTool):
    name: str = "ollama_manager"
    description: str = "Manages local Ollama models: list, pull, delete, and inspect model info."
    args_schema: type[BaseModel] = OllamaToolInput

    def _run(self, action: str, model_name: str = "") -> dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(self._async_run(action, model_name))

    async def _async_run(self, action: str, model_name: str = "") -> dict[str, Any]:
        import httpx
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                if action == "list_models":
                    r = await client.get(f"{base_url}/api/tags")
                    return r.json()
                elif action == "pull_model":
                    r = await client.post(f"{base_url}/api/pull", json={"name": model_name})
                    return {"status": "pulled", "model": model_name}
                elif action == "model_info":
                    r = await client.post(f"{base_url}/api/show", json={"name": model_name})
                    return r.json()
                elif action == "delete_model":
                    r = await client.delete(f"{base_url}/api/delete", json={"name": model_name})
                    return {"status": "deleted", "model": model_name}
                else:
                    return {"error": f"Unknown action: {action}"}
            except httpx.ConnectError as exc:
                from src.utils.error_handling import OllamaConnectionError
                raise OllamaConnectionError(f"Cannot connect to Ollama at {base_url}") from exc