from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SkillLevel(IntEnum):
    NONE = 0
    BEGINNER = 1
    ELEMENTARY = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


class SkillEntry(BaseModel):
    name: str
    level: SkillLevel
    source: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    last_evidenced: datetime | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip().lower()


class GitHubSignals(BaseModel):
    languages: dict[str, float] = Field(default_factory=dict)
    frameworks: list[str] = Field(default_factory=list)
    contribution_streak_days: int = 0
    project_complexity_score: float = Field(ge=0, le=10, default=0)
    key_projects: list[dict[str, Any]] = Field(default_factory=list)
    collaboration_signals: dict[str, Any] = Field(default_factory=dict)
    raw_url: str = ""


class KaggleSignals(BaseModel):
    tier: str = "Novice"
    medals: dict[str, int] = Field(default_factory=dict)
    ml_domains: list[str] = Field(default_factory=list)
    notebook_quality_score: float = Field(ge=0, le=10, default=0)
    active_last_year: bool = False
    strongest_domain: str = ""


class SkillProfile(BaseModel):
    user_id: str
    target_role: str
    skills: list[SkillEntry] = Field(default_factory=list)
    github_signals: GitHubSignals | None = None
    kaggle_signals: KaggleSignals | None = None
    experience_years_by_domain: dict[str, float] = Field(default_factory=dict)
    certifications: list[str] = Field(default_factory=list)
    stated_career_goal: str | None = None
    education_summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_skill_level(self, skill_name: str) -> SkillLevel:
        normalized = skill_name.strip().lower()
        for s in self.skills:
            if s.name == normalized:
                return s.level
        return SkillLevel.NONE

    def top_skills(self, n: int = 10) -> list[SkillEntry]:
        return sorted(self.skills, key=lambda s: (s.level, s.confidence), reverse=True)[:n]