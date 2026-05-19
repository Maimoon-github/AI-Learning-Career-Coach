from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

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

    def _run(self, github_url: str, max_repos: int = 30) -> dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(
            self._async_run(github_url, max_repos)
        )

    async def _async_run(self, github_url: str, max_repos: int = 30) -> dict[str, Any]:
        try:
            from github import Github, GithubException

            token = os.getenv("GITHUB_TOKEN")
            g = Github(token)

            # Extract username from URL or use directly
            username = github_url.rstrip("/").split("/")[-1]
            if github_url.startswith("http"):
                username = github_url.split("github.com/")[-1].rstrip("/")

            user = g.get_user(username)
            repos = list(user.get_repos(type="owner", sort="updated"))[:max_repos]

            # Language aggregation
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

                # Framework detection from topics and description
                topics = repo.get_topics()
                frameworks.update(t for t in topics if t not in ("python", "javascript"))

                if repo.stargazers_count > 5 or (repo.size or 0) > 500:
                    key_projects.append({
                        "name": repo.name,
                        "description": repo.description or "",
                        "tech_stack": topics,
                        "stars": repo.stargazers_count,
                        "size_kb": repo.size,
                        "last_updated": repo.updated_at.isoformat() if repo.updated_at else None,
                    })

            total_bytes = sum(lang_bytes.values()) or 1
            language_percentages = {
                lang: round(count / total_bytes * 100, 1)
                for lang, count in sorted(lang_bytes.items(), key=lambda x: -x[1])
            }

            # Contribution streak (approximation via events)
            try:
                events = list(user.get_events())[:100]
                push_dates = sorted(
                    {e.created_at.date() for e in events if e.type == "PushEvent"},
                    reverse=True,
                )
                streak = 0
                if push_dates:
                    from datetime import date, timedelta
                    check_date = push_dates[0]
                    for d in push_dates:
                        if d >= check_date - timedelta(days=streak + 1):
                            streak += 1
                        else:
                            break
            except Exception:
                streak = 0

            complexity_score = min(10, len(key_projects) * 1.5 + len(language_percentages) * 0.5)

            log.info("github_analysis_complete", username=username, repos_analyzed=len(repos))
            return {
                "languages": language_percentages,
                "frameworks": sorted(frameworks),
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

        except ImportError:
            log.error("PyGithub not installed")
            return {"error": "PyGithub not available", "languages": {}, "frameworks": []}
        except Exception as exc:
            log.error("github_tool_error", error=str(exc), url=github_url)
            return {"error": str(exc), "languages": {}, "frameworks": []}