"""
github_tool.py
==============
Async-native CrewAI BaseTool for deep GitHub profile analysis.

Architecture
------------
* Transport  : httpx.AsyncClient → GitHub REST API v3 + GraphQL (no PyGithub)
* Auth       : GITHUB_TOKEN env var (optional; unauthenticated OK, lower rate limit)
* Caching    : module-level TTLCache (shared; thread-safe via Lock)
* Retries    : tenacity exponential back-off on 429 / 5xx
* Analysis   : radon (CC + MI), bandit (security), ruff subprocess check
* Sync compat: _run() bridges to asyncio.run() or ThreadPoolExecutor

Dependencies (add to requirements.txt)
---------------------------------------
    httpx>=0.27.0          # already present
    tenacity>=8.3.0        # already present
    cachetools>=5.3.3      # already present
    structlog>=24.1.0      # already present
    radon>=6.0.1           # python code metrics
    # githubkit>=0.12.0    # optional richer typed client (not required)
    # bandit>=1.8.0        # optional security scan (subprocess)
    # ruff>=0.4.0          # optional style/lint (subprocess)

Usage
-----
    >>> tool = GitHubTool()
    >>> result = tool._run("https://github.com/octocat")
    >>> result["project_complexity_score"]
    3.5
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
_API_BASE = "https://api.github.com"
_GRAPHQL_URL = "https://api.github.com/graphql"
_CACHE_TTL = 3600           # 1 hour
_CACHE_MAXSIZE = 128
_HTTP_TIMEOUT = 20.0        # seconds
_MAX_ANALYSIS_FILES = 5     # Python files to pull for radon/bandit analysis
_MAX_FILE_SIZE_KB = 64      # skip files larger than this

_LANG_FILTER = frozenset({  # noisy topics to skip from frameworks list
    "python", "javascript", "typescript", "java", "ruby", "go", "rust",
    "c", "cpp", "c-plus-plus", "c-sharp", "shell", "html", "css",
})

# ---------------------------------------------------------------------------
# Module-level cache (shared across instances; thread-safe)
# ---------------------------------------------------------------------------
_CACHE: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)
_CACHE_LOCK = Lock()

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class GitHubInput(BaseModel):
    """Input schema for GitHubTool."""

    github_url: str = Field(
        description="Full GitHub profile URL (https://github.com/username) or bare username."
    )
    max_repos: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum number of repositories to fetch and analyze.",
    )
    run_code_analysis: bool = Field(
        default=False,
        description=(
            "If True, downloads Python source samples from top repos and runs "
            "radon (CC + MI) and optional bandit security checks. Slower but richer."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers — GitHub URL / username parsing
# ---------------------------------------------------------------------------

_GH_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)"
)


def _parse_username(raw: str) -> str:
    """
    Extract a valid GitHub username from a URL or bare name.

    Raises ValueError if the input cannot yield a plausible username.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("github_url must not be empty")

    # Try URL pattern first
    m = _GH_URL_RE.search(raw)
    if m:
        return m.group(1)

    # Bare username — validate GitHub naming rules
    bare = raw.lstrip("@").split("/")[0]
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", bare):
        return bare

    raise ValueError(f"Cannot parse a valid GitHub username from: {raw!r}")


# ---------------------------------------------------------------------------
# Helpers — HTTP client factory
# ---------------------------------------------------------------------------

def _make_headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    hdrs = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AI-Learning-Career-Coach/2.0 (async-httpx)",
    }
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    return hdrs


# ---------------------------------------------------------------------------
# Retry decorator for transient HTTP errors
# ---------------------------------------------------------------------------

def _github_retry():
    return retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        reraise=True,
    )


# ---------------------------------------------------------------------------
# Code-quality analysis helpers (radon + bandit + ruff via subprocess)
# ---------------------------------------------------------------------------

def _radon_analyze(source: str) -> dict[str, Any]:
    """
    Run radon CC + MI on a source string.
    Returns averaged cyclomatic complexity and maintainability index.
    Falls back gracefully if radon is not installed.
    """
    try:
        from radon.complexity import cc_visit, average_complexity  # type: ignore
        from radon.metrics import mi_visit  # type: ignore

        blocks = cc_visit(source)
        avg_cc = average_complexity(blocks) if blocks else 1.0
        mi = mi_visit(source, multi=True)
        return {"avg_cyclomatic_complexity": round(avg_cc, 2), "maintainability_index": round(mi, 1)}
    except Exception as exc:
        log.debug("radon_unavailable", error=str(exc))
        return {}


def _bandit_scan(source: str) -> dict[str, Any]:
    """
    Write source to a temp file and run bandit for security issues.
    Returns summary counts; falls back if bandit is not installed.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(source)
            fname = f.name

        proc = subprocess.run(
            ["bandit", "-q", "-r", fname, "-f", "json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        import json as _json

        data = _json.loads(proc.stdout) if proc.stdout.strip() else {}
        metrics = data.get("metrics", {}).get(fname, {})
        return {
            "bandit_high": int(metrics.get("SEVERITY.HIGH", 0)),
            "bandit_medium": int(metrics.get("SEVERITY.MEDIUM", 0)),
            "bandit_low": int(metrics.get("SEVERITY.LOW", 0)),
        }
    except FileNotFoundError:
        log.debug("bandit_not_installed")
        return {}
    except Exception as exc:
        log.debug("bandit_error", error=str(exc))
        return {}
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


def _ruff_check(source: str) -> int:
    """
    Run ruff on source, return number of lint violations.
    Returns -1 if ruff is not available.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(source)
            fname = f.name

        proc = subprocess.run(
            ["ruff", "check", "--output-format", "json", fname],
            capture_output=True,
            text=True,
            timeout=10,
        )
        import json as _json

        items = _json.loads(proc.stdout) if proc.stdout.strip() else []
        return len(items)
    except FileNotFoundError:
        return -1
    except Exception:
        return -1
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Core async analysis logic
# ---------------------------------------------------------------------------

async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    label: str = "",
) -> Any:
    """GET a JSON endpoint with retry, returning parsed body or {}."""
    @_github_retry()
    async def _call() -> Any:
        r = await client.get(url, params=params)
        if r.status_code == 404:
            return {}
        if r.status_code == 403:
            reset = r.headers.get("x-ratelimit-reset", "")
            log.warning("github_rate_limited", reset_at=reset, url=url)
            # Don't retry on 403 to avoid hammering
            return {}
        r.raise_for_status()
        return r.json()

    try:
        return await _call()
    except (RetryError, Exception) as exc:
        log.warning("github_fetch_failed", url=url, label=label, error=str(exc))
        return {}


async def _analyze_profile(
    username: str,
    max_repos: int,
    run_code_analysis: bool,
) -> dict[str, Any]:
    """
    Full async analysis of a GitHub profile. Returns the result dict.
    """
    headers = _make_headers()

    async with httpx.AsyncClient(headers=headers, timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        # --- 1. User metadata ---
        user: dict = await _fetch_json(client, f"{_API_BASE}/users/{username}", label="user")
        if not user or "login" not in user:
            return {"error": f"User '{username}' not found on GitHub.", "languages": {}, "frameworks": []}

        # --- 2. Repositories (paginated, up to max_repos) ---
        repos: list[dict] = []
        page = 1
        while len(repos) < max_repos:
            batch: Any = await _fetch_json(
                client,
                f"{_API_BASE}/users/{username}/repos",
                params={"type": "owner", "sort": "updated", "per_page": min(100, max_repos - len(repos)), "page": page},
                label=f"repos_page_{page}",
            )
            if not isinstance(batch, list) or not batch:
                break
            repos.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        # Filter out forks
        owned_repos = [r for r in repos if not r.get("fork", True)]

        # --- 3. Language aggregation (parallel) ---
        lang_futures = [
            _fetch_json(client, r["languages_url"], label=f"langs_{r['name']}")
            for r in owned_repos
        ]
        lang_results = await asyncio.gather(*lang_futures, return_exceptions=True)

        lang_bytes: dict[str, int] = {}
        for result in lang_results:
            if isinstance(result, dict):
                for lang, count in result.items():
                    lang_bytes[lang] = lang_bytes.get(lang, 0) + count

        total_bytes = sum(lang_bytes.values()) or 1
        language_percentages = {
            lang: round(count / total_bytes * 100, 1)
            for lang, count in sorted(lang_bytes.items(), key=lambda x: -x[1])
        }

        # --- 4. Topics → frameworks ---
        frameworks: set[str] = set()
        for r in owned_repos:
            for topic in r.get("topics", []):
                if topic.lower() not in _LANG_FILTER:
                    frameworks.add(topic)

        # --- 5. Key projects ---
        key_projects = []
        for r in sorted(owned_repos, key=lambda x: (x.get("stargazers_count", 0), x.get("size", 0)), reverse=True):
            stars = r.get("stargazers_count", 0)
            size_kb = r.get("size", 0)
            if stars >= 3 or size_kb >= 300:
                key_projects.append({
                    "name": r["name"],
                    "description": (r.get("description") or "")[:200],
                    "tech_stack": r.get("topics", [])[:10],
                    "stars": stars,
                    "forks": r.get("forks_count", 0),
                    "size_kb": size_kb,
                    "language": r.get("language") or "",
                    "open_issues": r.get("open_issues_count", 0),
                    "last_updated": r.get("updated_at", ""),
                    "url": r.get("html_url", ""),
                })

        # --- 6. Contribution streak (via events) ---
        streak = 0
        activity_days_last_year = 0
        try:
            events: Any = await _fetch_json(
                client,
                f"{_API_BASE}/users/{username}/events/public",
                params={"per_page": 100},
                label="events",
            )
            if isinstance(events, list):
                push_dates = sorted(
                    {
                        datetime.fromisoformat(e["created_at"].replace("Z", "+00:00")).date()
                        for e in events
                        if e.get("type") == "PushEvent" and e.get("created_at")
                    },
                    reverse=True,
                )
                if push_dates:
                    from datetime import date, timedelta

                    streak = 1
                    prev = push_dates[0]
                    for d in push_dates[1:]:
                        if (prev - d).days <= 1:
                            streak += 1
                            prev = d
                        else:
                            break

                    cutoff = datetime.now(timezone.utc).date() - timedelta(days=365)
                    activity_days_last_year = sum(1 for d in push_dates if d >= cutoff)
        except Exception as exc:
            log.debug("events_analysis_failed", error=str(exc))

        # --- 7. Project complexity score (heuristic + optional radon) ---
        radon_metrics: dict[str, Any] = {}
        bandit_metrics: dict[str, Any] = {}
        ruff_violations: int = -1

        if run_code_analysis:
            # Find Python files in top starred repos and analyse them
            python_repos = [r for r in owned_repos if r.get("language") == "Python"][:3]
            all_sources: list[str] = []

            for pr in python_repos:
                # Get root tree
                branch = pr.get("default_branch", "main")
                tree: Any = await _fetch_json(
                    client,
                    f"{_API_BASE}/repos/{username}/{pr['name']}/git/trees/{branch}",
                    params={"recursive": "1"},
                    label=f"tree_{pr['name']}",
                )
                py_blobs = [
                    item for item in (tree.get("tree") or [])
                    if item.get("path", "").endswith(".py")
                    and (item.get("size") or 0) < _MAX_FILE_SIZE_KB * 1024
                ][:_MAX_ANALYSIS_FILES]

                for blob in py_blobs:
                    raw: Any = await _fetch_json(
                        client,
                        f"{_API_BASE}/repos/{username}/{pr['name']}/contents/{blob['path']}",
                        label=f"file_{blob['path']}",
                    )
                    if isinstance(raw, dict) and raw.get("encoding") == "base64":
                        import base64

                        try:
                            source = base64.b64decode(raw["content"]).decode("utf-8", errors="replace")
                            all_sources.append(source)
                        except Exception:
                            pass

            if all_sources:
                combined = "\n\n".join(all_sources[:_MAX_ANALYSIS_FILES])
                radon_metrics = _radon_analyze(combined)
                bandit_metrics = _bandit_scan(combined)
                ruff_violations = _ruff_check(combined)

        # Complexity score: blend of project count, language diversity, stars, radon
        base_score = min(8.0, len(key_projects) * 1.2 + len(language_percentages) * 0.4)
        if radon_metrics.get("avg_cyclomatic_complexity"):
            cc = radon_metrics["avg_cyclomatic_complexity"]
            # Higher CC → slightly higher complexity score (capped contribution)
            base_score = min(10.0, base_score + min(2.0, cc * 0.2))
        complexity_score = round(base_score, 1)

        # --- 8. Collaboration signals ---
        collab = {
            "public_repos": user.get("public_repos", 0),
            "public_gists": user.get("public_gists", 0),
            "followers": user.get("followers", 0),
            "following": user.get("following", 0),
            "total_stars_earned": sum(r.get("stargazers_count", 0) for r in owned_repos),
            "total_forks_earned": sum(r.get("forks_count", 0) for r in owned_repos),
            "hireable": user.get("hireable"),
            "company": user.get("company") or "",
            "location": user.get("location") or "",
            "blog": user.get("blog") or "",
            "twitter_username": user.get("twitter_username") or "",
        }

        # --- 9. Assemble result ---
        result: dict[str, Any] = {
            # Core fields (backward-compatible)
            "languages": language_percentages,
            "frameworks": sorted(frameworks)[:25],
            "contribution_streak_days": streak,
            "project_complexity_score": complexity_score,
            "key_projects": key_projects[:10],
            "collaboration_signals": collab,
            "raw_url": f"https://github.com/{username}",
            # Enhanced fields
            "username": username,
            "name": user.get("name") or username,
            "bio": user.get("bio") or "",
            "account_created": user.get("created_at", ""),
            "activity_days_last_year": activity_days_last_year,
            "total_repos_analyzed": len(owned_repos),
            "top_language": next(iter(language_percentages), ""),
        }

        if radon_metrics:
            result["code_quality_metrics"] = {
                **radon_metrics,
                **({"ruff_violations": ruff_violations} if ruff_violations >= 0 else {}),
            }
        if bandit_metrics:
            result["security_metrics"] = bandit_metrics

        return result


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------

class GitHubTool(BaseTool):
    """
    Async-native CrewAI tool that performs deep analysis of a GitHub profile.

    Capabilities
    ------------
    - Language breakdown (% of total code by bytes)
    - Framework / topic detection
    - Contribution streak and activity metrics
    - Key project listing with metadata
    - Collaboration signals (followers, stars earned, etc.)
    - Optional code-quality analysis via radon/bandit/ruff

    Authentication
    --------------
    Set GITHUB_TOKEN environment variable for authenticated requests
    (5 000 req/hr vs 60 req/hr unauthenticated).

    Example
    -------
    >>> tool = GitHubTool()
    >>> data = tool._run("https://github.com/octocat")
    >>> data["top_language"]
    'Ruby'
    """

    name: str = "github_profile_analyzer"
    description: str = (
        "Analyzes a GitHub profile and returns programming languages, frameworks, "
        "top projects, contribution streak, and collaboration signals. "
        "Optionally runs code-quality checks (radon CC/MI, bandit security, ruff lint) "
        "on Python repositories. Pass a full GitHub URL or bare username."
    )
    args_schema: type[BaseModel] = GitHubInput

    # -----------------------------------------------------------------
    # Sync entry point (CrewAI calls _run)
    # -----------------------------------------------------------------
    def _run(  # type: ignore[override]
        self,
        github_url: str,
        max_repos: int = 30,
        run_code_analysis: bool = False,
    ) -> dict[str, Any]:
        """
        Sync wrapper — bridges to _async_run via asyncio.

        Handles both "already in event loop" (uses thread pool) and
        "no event loop" (creates one) scenarios correctly.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context (e.g., Jupyter, FastAPI, LangGraph)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._async_run(github_url, max_repos, run_code_analysis),
                )
                return future.result()
        else:
            return asyncio.run(self._async_run(github_url, max_repos, run_code_analysis))

    # -----------------------------------------------------------------
    # Async core
    # -----------------------------------------------------------------
    async def _async_run(
        self,
        github_url: str,
        max_repos: int = 30,
        run_code_analysis: bool = False,
    ) -> dict[str, Any]:
        """
        Fully async profile analysis with caching.

        Returns a dict with keys: languages, frameworks, contribution_streak_days,
        project_complexity_score, key_projects, collaboration_signals, raw_url,
        and optionally: code_quality_metrics, security_metrics.
        """
        # --- Parse username ---
        try:
            username = _parse_username(github_url)
        except ValueError as exc:
            log.error("github_username_parse_error", raw=github_url, error=str(exc))
            return {"error": str(exc), "languages": {}, "frameworks": []}

        # --- Cache lookup ---
        cache_key = hashlib.sha256(
            f"{username}:{max_repos}:{run_code_analysis}".encode()
        ).hexdigest()[:16]

        with _CACHE_LOCK:
            if cache_key in _CACHE:
                log.debug("github_cache_hit", username=username)
                return _CACHE[cache_key]

        log.info("github_analysis_start", username=username, max_repos=max_repos, code_analysis=run_code_analysis)

        try:
            result = await _analyze_profile(username, max_repos, run_code_analysis)
        except Exception as exc:
            log.error("github_analysis_error", username=username, error=str(exc), exc_info=True)
            return {
                "error": str(exc),
                "languages": {},
                "frameworks": [],
                "raw_url": f"https://github.com/{username}",
            }

        log.info(
            "github_analysis_complete",
            username=username,
            top_lang=result.get("top_language"),
            score=result.get("project_complexity_score"),
        )

        with _CACHE_LOCK:
            _CACHE[cache_key] = result

        return result


# ---------------------------------------------------------------------------
# Module-level convenience (for direct script execution / smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tool = GitHubTool()
    # Smoke test against the public octocat profile (no auth required)
    data = tool._run("https://github.com/octocat", max_repos=10)
    print(json.dumps(data, indent=2, default=str))