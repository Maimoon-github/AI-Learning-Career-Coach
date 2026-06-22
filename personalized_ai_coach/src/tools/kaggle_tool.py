"""
kaggle_tool.py
==============
Async-ready CrewAI BaseTool for deep Kaggle profile analysis.

Architecture
------------
* Auth       : KAGGLE_USERNAME + KAGGLE_KEY env vars  OR  ~/.kaggle/kaggle.json
* API client : kaggle.api.kaggle_api_extended.KaggleApiExtended (v2.2+)
* Notebook   : kernels_list(user=<username>) → optional nbformat cell analysis
* Caching    : module-level TTLCache (thread-safe)
* Sync compat: _run() bridges to asyncio via ThreadPoolExecutor when inside
               a running event loop (LangGraph / FastAPI / Jupyter safe).

Dependencies
------------
    kaggle>=2.2.0        # official Kaggle SDK (already installed)
    nbformat>=5.10.0     # notebook parsing (already installed)
    structlog>=24.1.0    # structured logging (already installed)
    cachetools>=5.3.0    # TTL cache (already installed)
    httpx>=0.27.0        # public Kaggle page scraping fallback (already installed)

Authentication
--------------
Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, OR place a valid
kaggle.json at ~/.kaggle/kaggle.json.  Without credentials the tool returns
a graceful degradation result.

Example
-------
>>> tool = KaggleTool()
>>> result = tool._run("abhishek")
>>> result["tier"]
'Grandmaster'

>>> # Quick smoke test (public profile, no auth required for kernel listing
>>> # with a valid API key):
>>> result = asyncio.run(KaggleTool()._async_run("zackoneil"))
>>> print(result.get("strongest_domain"))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import structlog
from cachetools import TTLCache
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CACHE_TTL = 3600          # seconds
_CACHE_MAXSIZE = 64
_MAX_NOTEBOOKS = 10        # notebooks to inspect per user
_MAX_NOTEBOOK_CELLS = 50   # cells to analyse per notebook
_HTTP_TIMEOUT = 15.0

# Kaggle Tier thresholds (points-based when available, else heuristic fallback)
_TIER_THRESHOLDS = {
    "Grandmaster": {"competitions": 5, "solo_gold": 1, "gold": 5},
    "Master":      {"competitions": 3, "gold": 1, "silver_or_gold": 3},
    "Expert":      {"competitions": 1, "bronze_or_better": 5},
    "Contributor": {"competitions": 0, "notebooks": 1},
    "Novice":      {},
}

# Domain keyword map for competition title → ML domain
_DOMAIN_MAP: dict[str, list[str]] = {
    "nlp":             ["nlp", "text", "language", "sentiment", "toxic", "bert", "llm",
                        "translation", "summarization", "squad", "ner", "qa"],
    "computer_vision": ["image", "vision", "detection", "segmentation", "classification",
                        "yolo", "cnn", "object", "face", "ocr", "optical"],
    "tabular":         ["tabular", "regression", "classification", "feature", "prediction",
                        "house", "price", "churn", "fraud", "credit", "sales"],
    "time_series":     ["time-series", "forecast", "temporal", "stock", "demand", "energy",
                        "weather", "signal"],
    "audio":           ["audio", "speech", "sound", "voice", "asr", "music"],
    "reinforcement":   ["rl", "reinforcement", "agent", "game", "atari", "simulation"],
    "generative":      ["generative", "gan", "diffusion", "stable-diffusion", "dalle",
                        "image-generation", "synthetic"],
}

# Programming language → notebook quality factor
_LANG_QUALITY_WEIGHT = {"python": 1.0, "r": 0.9, "sql": 0.7, "julia": 0.8}

# ---------------------------------------------------------------------------
# Module-level TTL cache
# ---------------------------------------------------------------------------
_CACHE: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)
_CACHE_LOCK = Lock()


# ---------------------------------------------------------------------------
# Username validation
# ---------------------------------------------------------------------------
_KAGGLE_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{1,38}[A-Za-z0-9]$", re.ASCII)


def _validate_username(raw: str) -> str:
    """
    Normalise and validate a Kaggle username.

    Accepts bare usernames and profile URLs like:
        https://www.kaggle.com/abhishek
    Raises ValueError on invalid input.
    """
    raw = raw.strip().rstrip("/")
    if not raw:
        raise ValueError("Kaggle username must not be empty.")

    # Strip URL prefix if present
    for prefix in ("https://www.kaggle.com/", "http://www.kaggle.com/", "kaggle.com/"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):]

    # Take only the first path segment (ignore /code, /datasets, etc.)
    username = raw.split("/")[0].strip("@")

    if not _KAGGLE_USERNAME_RE.fullmatch(username) and len(username) >= 2:
        # Allow single-char or short usernames too (relax regex slightly)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", username):
            raise ValueError(f"Cannot parse a valid Kaggle username from: {raw!r}")

    return username


# ---------------------------------------------------------------------------
# Kaggle API helpers
# ---------------------------------------------------------------------------

def _get_api():
    """
    Authenticate and return a KaggleApiExtended instance.

    The Kaggle SDK v2.2 prints an auth-help banner and calls exit(1) when
    no credentials are present. We silence the banner by temporarily replacing
    sys.stdout/sys.stderr, then catch SystemExit to raise a clear RuntimeError.
    """
    import builtins, io, sys as _sys  # noqa: PLC0415

    class _Sink(io.StringIO):
        """Discards all writes."""
        def write(self, s: str) -> int:   # type: ignore[override]
            return len(s)
        def flush(self) -> None:
            pass

    _sink = _Sink()
    _old_stdout = _sys.stdout
    _old_stderr = _sys.stderr
    _old_print  = builtins.print

    def _noop(*a, **kw): pass  # noqa: E704

    try:
        _sys.stdout  = _sink
        _sys.stderr  = _sink
        builtins.print = _noop
        from kaggle.api.kaggle_api_extended import KaggleApiExtended  # noqa: PLC0415
        api = KaggleApiExtended()
        api.authenticate()
        return api
    except SystemExit:
        raise RuntimeError(
            "No Kaggle credentials found. Set KAGGLE_USERNAME + KAGGLE_KEY "
            "environment variables, or place ~/.kaggle/kaggle.json."
        )
    finally:
        _sys.stdout    = _old_stdout
        _sys.stderr    = _old_stderr
        builtins.print = _old_print


def _safe_attr(obj: Any, *attrs: str, default: Any = None) -> Any:
    """Safely traverse a chain of attributes; return default on any failure."""
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return default
    return obj if obj is not None else default


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

def _detect_domains(titles: list[str]) -> dict[str, int]:
    """
    Count domain hits from a list of competition/notebook titles.
    Returns {domain: hit_count} dict sorted by count descending.
    """
    hits: dict[str, int] = {d: 0 for d in _DOMAIN_MAP}
    for title in titles:
        t_lower = title.lower()
        for domain, keywords in _DOMAIN_MAP.items():
            if any(kw in t_lower for kw in keywords):
                hits[domain] += 1
    return dict(sorted(hits.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Tier estimation
# ---------------------------------------------------------------------------

def _estimate_tier(
    *,
    competition_count: int,
    notebook_count: int,
    medals: dict[str, int],
) -> str:
    """
    Estimate Kaggle tier from observable signals.

    Kaggle's actual ranking system is opaque; we use a conservative
    heuristic that tends to under-estimate rather than over-estimate.

    Thresholds (approximate, not official):
        Grandmaster : ≥5 competitions + any solo gold  OR ≥5 golds
        Master      : ≥3 competitions + ≥1 gold         OR ≥3 gold/silver
        Expert      : ≥1 competition  + ≥5 bronze+        OR ≥10 notebooks
        Contributor : has at least submitted once
        Novice      : no submissions detected
    """
    gold   = medals.get("gold", 0)
    silver = medals.get("silver", 0)
    bronze = medals.get("bronze", 0)
    total_medals = gold + silver + bronze

    if competition_count >= 5 and (gold >= 5):
        return "Grandmaster"
    if competition_count >= 3 and gold >= 1:
        return "Master"
    if competition_count >= 1 and (gold + silver) >= 3:
        return "Master"
    if competition_count >= 1 and total_medals >= 5:
        return "Expert"
    if notebook_count >= 10 and total_medals >= 2:
        return "Expert"
    if competition_count >= 1 or notebook_count >= 1:
        return "Contributor"
    return "Novice"


# ---------------------------------------------------------------------------
# Notebook quality analysis (nbformat)
# ---------------------------------------------------------------------------

def _analyse_notebook_source(source: str) -> dict[str, Any]:
    """
    Parse a .ipynb JSON string with nbformat and extract quality signals.

    Returns a dict with:
        total_cells, code_cells, markdown_cells, total_loc,
        has_visualisation, has_modelling, uses_imports (list), quality_score
    """
    try:
        import nbformat  # noqa: PLC0415

        nb = nbformat.reads(source, as_version=4)
    except Exception:
        return {}

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_cells   = [c for c in nb.cells if c.cell_type == "markdown"]

    total_loc       = sum(len(c.source.splitlines()) for c in code_cells)
    all_code        = "\n".join(c.source for c in code_cells)

    # Detect visualisation
    viz_keywords   = {"matplotlib", "seaborn", "plotly", "bokeh", "altair",
                      "plt.show", "fig.show", "px.", "sns."}
    has_viz        = any(kw in all_code for kw in viz_keywords)

    # Detect ML model usage
    ml_keywords    = {"sklearn", "xgboost", "lightgbm", "catboost", "keras",
                      "tensorflow", "torch", "fit(", "predict(", "train_test_split"}
    has_ml         = any(kw in all_code for kw in ml_keywords)

    # Extract imported libraries
    import_re      = re.compile(r"^\s*(?:import|from)\s+([\w\.]+)", re.MULTILINE)
    imports        = sorted({m.group(1).split(".")[0] for m in import_re.finditer(all_code)})

    # Quality heuristic (0–10)
    score  = 0.0
    score += min(3.0, len(code_cells) * 0.3)        # code depth
    score += min(2.0, len(md_cells) * 0.4)           # documentation
    score += 1.5 if has_viz else 0.0                  # visualisation
    score += 2.0 if has_ml else 0.0                   # modelling
    score += min(1.5, total_loc / 200)                # code volume

    return {
        "total_cells":       len(nb.cells),
        "code_cells":        len(code_cells),
        "markdown_cells":    len(md_cells),
        "total_loc":         total_loc,
        "has_visualisation": has_viz,
        "has_modelling":     has_ml,
        "uses_imports":      imports[:15],
        "quality_score":     round(min(10.0, score), 1),
    }


# ---------------------------------------------------------------------------
# Kaggle public REST fallback (no auth needed for some endpoints)
# ---------------------------------------------------------------------------

async def _scrape_public_profile(username: str) -> dict[str, Any]:
    """
    Lightweight scrape of the Kaggle public profile page via httpx.
    Extracts tier badge text and any data-* attributes available without auth.
    Falls back silently on any error.
    """
    try:
        import httpx  # noqa: PLC0415

        url = f"https://www.kaggle.com/{username}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Coach/2.0)"}
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return {}
            html = resp.text

        # Look for tier badge text
        tier_match = re.search(
            r'"currentUrl":"/@?[\w-]+","tier":"?(\w+)"?', html
        )
        tier = tier_match.group(1).capitalize() if tier_match else ""

        # Total votes on notebooks
        votes_match = re.search(r'"totalVotes":(\d+)', html)
        total_votes = int(votes_match.group(1)) if votes_match else 0

        return {"scraped_tier": tier, "total_notebook_votes": total_votes}

    except Exception as exc:
        log.debug("kaggle_scrape_failed", error=str(exc))
        return {}


# ---------------------------------------------------------------------------
# Core analysis logic (runs in a thread to keep async loop unblocked)
# ---------------------------------------------------------------------------

def _analyse_profile_sync(username: str, analyse_notebooks: bool) -> dict[str, Any]:
    """
    Synchronous heavy-lifting: authenticates, calls Kaggle API, analyses
    notebooks.  Executed via executor so the event loop stays unblocked.
    """
    try:
        api = _get_api()
    except Exception as exc:
        log.warning("kaggle_auth_failed", error=str(exc))
        # Return degraded result — no credentials available
        return _degraded_result(username, error=f"Authentication failed: {exc}")

    # ── Competitions ────────────────────────────────────────────────────────
    comp_titles: list[str] = []
    medals: dict[str, int] = {"gold": 0, "silver": 0, "bronze": 0}

    try:
        # competitions_list with search returns competitions related to the username
        # (Kaggle API v2.2 doesn't have a direct "competitions by user" endpoint,
        # so we fetch a broad set and also use kernels_list for user notebooks)
        response = api.competitions_list(page_size=50, page=1)
        competitions_raw = getattr(response, "competitions", []) or []
        all_comps = list(competitions_raw)

        # Also search by username to get user-related competitions
        search_resp = api.competitions_list(search=username, page_size=50)
        search_comps = list(getattr(search_resp, "competitions", []) or [])

        # Merge, deduplicate by ref
        seen_refs: set[str] = set()
        for comp in all_comps + search_comps:
            ref = _safe_attr(comp, "ref", default="")
            if ref and ref not in seen_refs:
                seen_refs.add(ref)
                title = _safe_attr(comp, "title", default="")
                if title:
                    comp_titles.append(str(title))

            # Medal approximation from leaderboard position (if entered)
            if _safe_attr(comp, "userHasEntered", default=False):
                rank = _safe_attr(comp, "userRank", default=None)
                team_count = _safe_attr(comp, "teamCount", default=0) or 1
                if rank is not None:
                    pct = rank / team_count
                    if pct <= 0.10:
                        medals["bronze"] += 1
                    if pct <= 0.05:
                        medals["silver"] += 1
                    if pct <= 0.01:
                        medals["gold"] += 1

        comp_count = len([1 for c in (all_comps + search_comps)
                          if _safe_attr(c, "userHasEntered", default=False)])

    except Exception as exc:
        log.warning("kaggle_competitions_failed", error=str(exc))
        comp_count  = 0
        comp_titles = []

    log.debug("kaggle_competitions_fetched", count=comp_count, titles=len(comp_titles))

    # ── Kernels (Notebooks) ─────────────────────────────────────────────────
    kernel_titles:  list[str] = []
    notebook_votes: int = 0
    notebook_count: int = 0
    notebook_insights: list[dict[str, Any]] = []

    try:
        kernels = api.kernels_list(
            user=username,
            page_size=min(_MAX_NOTEBOOKS, 20),
            sort_by="votes",
        ) or []

        notebook_count = len(kernels)
        for k in kernels:
            title = _safe_attr(k, "title", default="")
            votes = _safe_attr(k, "totalVotes", default=0) or 0
            notebook_votes += int(votes)
            if title:
                kernel_titles.append(str(title))

        log.debug("kaggle_kernels_fetched", count=notebook_count, votes=notebook_votes)

        # ── Per-notebook quality analysis ────────────────────────────────
        if analyse_notebooks and kernels:
            with tempfile.TemporaryDirectory(prefix="kaggle_nb_") as tmp_dir:
                for k in kernels[:3]:  # limit to top-3 by votes
                    kernel_ref = _safe_attr(k, "ref", default="")
                    if not kernel_ref:
                        continue
                    try:
                        # Pull notebook source into temp dir
                        api.kernels_pull(
                            kernel=kernel_ref,
                            path=tmp_dir,
                            metadata=False,
                            quiet=True,
                        )
                        import glob
                        nb_files = glob.glob(f"{tmp_dir}/*.ipynb")
                        if nb_files:
                            nb_source = open(nb_files[0], encoding="utf-8").read()
                            insight = _analyse_notebook_source(nb_source)
                            if insight:
                                insight["kernel_ref"] = kernel_ref
                                insight["votes"] = int(_safe_attr(k, "totalVotes", default=0))
                                notebook_insights.append(insight)
                    except Exception as nb_exc:
                        log.debug("kaggle_notebook_pull_failed",
                                  kernel=kernel_ref, error=str(nb_exc))

    except Exception as exc:
        log.warning("kaggle_kernels_failed", error=str(exc))

    # ── Domain detection ────────────────────────────────────────────────────
    all_titles   = comp_titles + kernel_titles
    domain_hits  = _detect_domains(all_titles)
    ml_domains   = [d for d, cnt in domain_hits.items() if cnt > 0]

    # ── Tier estimation ─────────────────────────────────────────────────────
    tier = _estimate_tier(
        competition_count=comp_count,
        notebook_count=notebook_count,
        medals=medals,
    )

    # ── Notebook quality score (average of analysed notebooks or heuristic) ─
    if notebook_insights:
        avg_nb_score = round(
            sum(n.get("quality_score", 0) for n in notebook_insights) / len(notebook_insights),
            1,
        )
    else:
        # Heuristic: votes / notebooks gives engagement proxy, cap at 10
        engagement  = min(1.0, notebook_votes / max(1, notebook_count * 10))
        avg_nb_score = round(min(10.0, engagement * 7.0 + notebook_count * 0.3), 1)

    # ── Activity signal ─────────────────────────────────────────────────────
    # "active last year" = at least one kernel recently edited
    recent_activity = False
    try:
        kernels_list_2 = api.kernels_list(user=username, page_size=5, sort_by="hotness") or []
        for k in kernels_list_2:
            last_run = _safe_attr(k, "lastRunTime", default=None)
            if last_run:
                from datetime import datetime, timezone  # noqa: PLC0415
                if isinstance(last_run, str):
                    last_run = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
                delta = datetime.now(timezone.utc) - last_run.replace(tzinfo=timezone.utc)
                if delta.days <= 365:
                    recent_activity = True
                    break
    except Exception:
        recent_activity = comp_count > 0 or notebook_count > 0

    # ── Strongest domain ────────────────────────────────────────────────────
    strongest = ml_domains[0] if ml_domains else "tabular"

    result: dict[str, Any] = {
        # ── Backward-compatible core fields ──────────────────────────────
        "tier":                     tier,
        "medals":                   medals,
        "ml_domains":               ml_domains,
        "notebook_quality_score":   avg_nb_score,
        "active_last_year":         recent_activity,
        "strongest_domain":         strongest,
        # ── Enhanced fields ───────────────────────────────────────────────
        "username":                 username,
        "competition_count":        comp_count,
        "notebook_count":           notebook_count,
        "total_notebook_votes":     notebook_votes,
        "domain_hit_counts":        {d: c for d, c in domain_hits.items() if c > 0},
        "top_competition_titles":   comp_titles[:5],
        "top_notebook_titles":      kernel_titles[:5],
        "notebook_insights":        notebook_insights,          # per-notebook quality
        "profile_url":              f"https://www.kaggle.com/{username}",
    }

    log.info(
        "kaggle_analysis_complete",
        username=username,
        tier=tier,
        competitions=comp_count,
        notebooks=notebook_count,
        domains=ml_domains[:3],
    )
    return result


# ---------------------------------------------------------------------------
# Graceful degradation factory
# ---------------------------------------------------------------------------

def _degraded_result(username: str, error: str = "") -> dict[str, Any]:
    """Return a safe, schema-consistent result when analysis fails."""
    return {
        "tier":                   "Novice",
        "medals":                 {"gold": 0, "silver": 0, "bronze": 0},
        "ml_domains":             [],
        "notebook_quality_score": 0.0,
        "active_last_year":       False,
        "strongest_domain":       "",
        "username":               username,
        "competition_count":      0,
        "notebook_count":         0,
        "total_notebook_votes":   0,
        "domain_hit_counts":      {},
        "top_competition_titles": [],
        "top_notebook_titles":    [],
        "notebook_insights":      [],
        "profile_url":            f"https://www.kaggle.com/{username}",
        **({"error": error} if error else {}),
    }


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class KaggleInput(BaseModel):
    """Input schema for KaggleTool."""

    username: str = Field(
        description=(
            "Kaggle username or full profile URL "
            "(e.g. 'abhishek' or 'https://www.kaggle.com/abhishek')."
        )
    )
    analyse_notebooks: bool = Field(
        default=False,
        description=(
            "If True, download and analyse the user's top notebooks via nbformat. "
            "Adds meaningful quality metrics but requires auth and is slower (~5-15s extra)."
        ),
    )


# ---------------------------------------------------------------------------
# CrewAI Tool
# ---------------------------------------------------------------------------

class KaggleTool(BaseTool):
    """
    CrewAI BaseTool that performs deep analysis of a Kaggle user profile.

    Capabilities
    ------------
    - Competition participation count and domain distribution
    - Medal estimation (gold/silver/bronze) from leaderboard ranks
    - Tier estimation (Novice → Grandmaster) via multi-signal heuristic
    - Top notebooks by votes with quality scores (nbformat analysis)
    - Activity signal (recent kernel activity within last year)
    - ML domain fingerprint across NLP, CV, tabular, time-series, audio, RL

    Authentication
    --------------
    Set KAGGLE_USERNAME + KAGGLE_KEY environment variables, or place
    ~/.kaggle/kaggle.json.  Without credentials the tool gracefully degrades.

    Example
    -------
    >>> tool = KaggleTool()
    >>> result = tool._run("abhishek")
    >>> result["tier"], result["strongest_domain"]
    ('Grandmaster', 'nlp')
    """

    name: str = "kaggle_profile_analyzer"
    description: str = (
        "Analyses a Kaggle user profile. Returns competition tier, medal counts, "
        "ML domain expertise, notebook quality scores, and activity signals. "
        "Pass a Kaggle username or full profile URL. "
        "Set analyse_notebooks=True for deeper notebook quality analysis."
    )
    args_schema: type[BaseModel] = KaggleInput

    # ── Sync entry point (CrewAI calls _run) ────────────────────────────────

    def _run(  # type: ignore[override]
        self,
        username: str,
        analyse_notebooks: bool = False,
    ) -> dict[str, Any]:
        """
        Sync wrapper – bridges to _async_run.

        Handles both "already inside a running event loop" (Jupyter/LangGraph)
        and "no event loop" (plain scripts) scenarios correctly.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside async context  → use a fresh thread
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self._async_run(username, analyse_notebooks)
                ).result()
        else:
            return asyncio.run(self._async_run(username, analyse_notebooks))

    # ── Async core ───────────────────────────────────────────────────────────

    async def _async_run(
        self,
        username: str,
        analyse_notebooks: bool = False,
    ) -> dict[str, Any]:
        """
        Async analysis with TTL caching.

        Runs the heavy synchronous Kaggle SDK calls in the default executor so
        the event loop stays responsive.  Optionally fires a lightweight public
        page scrape in parallel for supplemental data.
        """
        # ── Validate username ────────────────────────────────────────────────
        try:
            username = _validate_username(username)
        except ValueError as exc:
            log.error("kaggle_username_invalid", raw=username, error=str(exc))
            return _degraded_result(username, error=str(exc))

        # ── Cache check ──────────────────────────────────────────────────────
        cache_key = hashlib.sha256(
            f"{username}:{analyse_notebooks}".encode()
        ).hexdigest()[:16]

        with _CACHE_LOCK:
            if cache_key in _CACHE:
                log.debug("kaggle_cache_hit", username=username)
                return _CACHE[cache_key]

        log.info("kaggle_analysis_start", username=username, analyse_notebooks=analyse_notebooks)

        # ── Run sync analysis in executor ────────────────────────────────────
        loop = asyncio.get_event_loop()
        gathered = await asyncio.gather(
            loop.run_in_executor(
                None, _analyse_profile_sync, username, analyse_notebooks
            ),
            _scrape_public_profile(username),
            return_exceptions=True,
        )
        result, scrape = gathered[0], gathered[1]

        # Handle executor exceptions
        if isinstance(result, BaseException):
            log.error("kaggle_analysis_error", username=username, error=str(result))
            return _degraded_result(username, error=str(result))

        # Merge optional scrape results
        if isinstance(scrape, dict):
            # Prefer scraped tier if we got one and our heuristic says Novice
            scraped_tier = scrape.get("scraped_tier", "")
            if scraped_tier and result.get("tier") == "Novice":
                result["tier"] = scraped_tier
            if scrape.get("total_notebook_votes", 0) > result.get("total_notebook_votes", 0):
                result["total_notebook_votes"] = scrape["total_notebook_votes"]

        # ── Store in cache ───────────────────────────────────────────────────
        with _CACHE_LOCK:
            _CACHE[cache_key] = result  # type: ignore[assignment]

        return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Module-level convenience entry point (smoke test / direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke test:
        python -m personalized_ai_coach.src.tools.kaggle_tool
    or:
        python kaggle_tool.py

    Requires KAGGLE_USERNAME + KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).
    Tests against the public "kaggle" official account and "abhishek".
    """
    import sys

    test_users = sys.argv[1:] or ["kaggle", "abhishek"]

    async def _main():
        tool = KaggleTool()
        for user in test_users:
            print(f"\n{'='*60}\nAnalysing: {user}\n{'='*60}")
            result = await tool._async_run(user, analyse_notebooks=False)
            # Pretty-print without notebook_insights body for readability
            display = {k: v for k, v in result.items() if k != "notebook_insights"}
            print(json.dumps(display, indent=2, default=str))
            nb_insights = result.get("notebook_insights", [])
            if nb_insights:
                print(f"\nNotebook insights ({len(nb_insights)} analysed):")
                for nb in nb_insights:
                    print(f"  [{nb.get('kernel_ref')}] "
                          f"score={nb.get('quality_score')} "
                          f"cells={nb.get('total_cells')} "
                          f"viz={nb.get('has_visualisation')} "
                          f"ml={nb.get('has_modelling')}")

    asyncio.run(_main())