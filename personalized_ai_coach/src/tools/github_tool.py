from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta
from typing import Any

import structlog
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from cachetools import TTLCache
import hashlib

log = structlog.get_logger(__name__)

class GitHubInput(BaseModel):
    github_url: str = Field(description="Full GitHub profile URL or username")
    max_repos: int = Field(default=30, description="Max repos to analyze")

class GitHubTool(BaseTool):
    name: str = "github_profile_analyzer"
    description: str = (
        "Fetches and analyzes a GitHub profile. Returns languages, frameworks, "
        "contribution history, top repositories, and collaboration signals."
    )
    args_schema: type[BaseModel] = GitHubInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache = TTLCache(maxsize=100, ttl=3600)

    def _run(self, github_url: str, max_repos: int = 30) -> dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(self._async_run(github_url, max_repos))).result()
        else:
            return loop.run_until_complete(self._async_run(github_url, max_repos))

    async def _async_run(self, github_url: str, max_repos: int = 30) -> dict[str, Any]:
        cache_key = hashlib.md5(f"{github_url}:{max_repos}".encode()).hexdigest()
        if hasattr(self, "_cache") and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            from github import Github, GithubException

            token = os.getenv("GITHUB_TOKEN")
            g = Github(token) if token else Github()

            username = github_url.rstrip("/").split("/")[-1]
            if "github.com/" in github_url:
                username = github_url.split("github.com/")[-1].rstrip("/")

            user = g.get_user(username)
            repos = list(user.get_repos(type="owner", sort="updated"))[:max_repos]

            lang_bytes: dict[str, int] = {}
            frameworks: set[str] = set()
            key_projects = []

            for repo in repos:
                if repo.fork:
                    continue
                try:
                    langs = repo.get_languages()
                    for lang, count in langs.items():
                        lang_bytes[lang] = lang_bytes.get(lang, 0) + count
                except GithubException:
                    pass

                topics = repo.get_topics()
                frameworks.update(t for t in topics if t not in ("python", "javascript"))

                if repo.stargazers_count > 5 or (repo.size or 0) > 500:
                    key_projects.append({
                        "name": repo.name,
                        "description": repo.description or "",
                        "tech_stack": list(topics)[:10],
                        "stars": repo.stargazers_count,
                        "size_kb": repo.size,
                        "last_updated": repo.updated_at.isoformat() if repo.updated_at else None,
                    })

            total_bytes = sum(lang_bytes.values()) or 1
            language_percentages = {
                lang: round(count / total_bytes * 100, 1)
                for lang, count in sorted(lang_bytes.items(), key=lambda x: -x[1])
            }

            try:
                events = list(user.get_events())[:100]
                push_dates = sorted(
                    {e.created_at.date() for e in events if e.type == "PushEvent"},
                    reverse=True,
                )
                streak = 0
                if push_dates:
                    prev_date = push_dates[0]
                    streak = 1
                    for d in push_dates[1:]:
                        if (prev_date - d).days <= 1:
                            streak += 1
                            prev_date = d
                        else:
                            break
            except Exception:
                streak = 0

            complexity_score = min(10, len(key_projects) * 1.5 + len(language_percentages) * 0.5)
            
            result = {
                "languages": language_percentages,
                "frameworks": sorted(frameworks)[:20],
                "contribution_streak_days": streak,
                "project_complexity_score": round(complexity_score, 1),
                "key_projects": key_projects[:10],
                "collaboration_signals": {
                    "public_repos": user.public_repos,
                    "followers": user.followers,
                    "following": user.following,
                },
                "raw_url": github_url,
            }
            if hasattr(self, "_cache"):
                self._cache[cache_key] = result
            return result

        except Exception as exc:
            log.error("github_tool_error", error=str(exc), url=github_url)
            return {"error": str(exc), "languages": {}, "frameworks": []}