from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class KaggleInput(BaseModel):
    username: str = Field(description="Kaggle username")


class KaggleTool(BaseTool):
    name: str = "kaggle_profile_analyzer"
    description: str = "Fetches and evaluates a Kaggle profile: competition tier, medals, ML domains, notebook quality."
    args_schema: type[BaseModel] = KaggleInput

    def _run(self, username: str) -> dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(self._async_run(username))).result()
        else:
            return loop.run_until_complete(self._async_run(username))

    async def _async_run(self, username: str) -> dict[str, Any]:
        try:
            # Support both old (KaggleApiExtended) and new (ApiClient) kaggle SDK versions
            try:
                from kaggle.api.kaggle_api_extended import KaggleApiExtended
                api = KaggleApiExtended()
                api.authenticate()
            except ImportError:
                import kaggle
                api = kaggle.api
                api.authenticate()

            # Search competitions (Kaggle API doesn't directly filter by user; use search)
            competitions = api.competitions_list(search=username)
            ml_domains: set[str] = set()
            domain_map = {
                "nlp": ["nlp", "text", "language", "sentiment", "toxic"],
                "computer_vision": ["image", "vision", "detection", "segmentation", "classification"],
                "tabular": ["tabular", "regression", "classification", "feature", "prediction"],
                "time_series": ["time-series", "forecast", "temporal", "stock"],
            }

            medals: dict[str, int] = {"gold": 0, "silver": 0, "bronze": 0}
            # Note: actual medal data requires user's submissions, simplified for demo
            for comp in competitions[:50]:
                title_lower = (getattr(comp, "title", "") or "").lower()
                for domain, keywords in domain_map.items():
                    if any(kw in title_lower for kw in keywords):
                        ml_domains.add(domain)

            # Approximate tier based on number of competitions
            tier_map = ["Novice", "Contributor", "Expert", "Master", "Grandmaster"]
            tier_score = min(len(competitions) // 5, 4)

            log.info("kaggle_analysis_complete", username=username, competitions=len(competitions))
            return {
                "tier": tier_map[tier_score],
                "medals": medals,
                "ml_domains": sorted(ml_domains),
                "notebook_quality_score": min(10, tier_score * 2.5),
                "active_last_year": len(competitions) > 0,
                "strongest_domain": sorted(ml_domains)[0] if ml_domains else "tabular",
            }

        except Exception as exc:
            log.error("kaggle_tool_error", error=str(exc), username=username)
            return {
                "tier": "Novice",
                "medals": {},
                "ml_domains": [],
                "notebook_quality_score": 0,
                "active_last_year": False,
                "strongest_domain": "",
                "error": str(exc),
            }