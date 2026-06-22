from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class SkillLevel(IntEnum):
    """Standardized proficiency levels for skill assessment."""

    NONE = 0
    BEGINNER = 1
    ELEMENTARY = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


class SkillEntry(BaseModel):
    """A single skill within a user's profile with verified evidence."""

    name: str = Field(..., description="The name of the skill (e.g., 'Python', 'SQL')")
    level: SkillLevel = Field(
        SkillLevel.NONE, description="The assessed proficiency level"
    )
    source: list[str] = Field(
        default_factory=list,
        description="Origin of the data (e.g., 'github', 'user_input', 'assessment')",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Assurance level of the skill assessment (0.0 to 1.0)",
    )
    last_evidenced: Optional[datetime] = Field(
        None, description="Timestamp of the most recent activity proving this skill"
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Ensures skill names are consistent for cross-referencing."""
        return v.strip().lower()


class GitHubSignals(BaseModel):
    """Heuristic data extracted from a user's GitHub activity."""

    languages: dict[str, float] = Field(
        default_factory=dict, description="Language distribution as percentages"
    )
    frameworks: list[str] = Field(
        default_factory=list, description="Detected frameworks based on project files"
    )
    contribution_streak_days: int = Field(
        0, description="Number of consecutive days with commits"
    )
    project_complexity_score: float = Field(
        ge=0, le=10, default=0, description="AI-calculated complexity of repositories"
    )
    key_projects: list[dict[str, Any]] = Field(
        default_factory=list, description="Summary of most relevant repositories"
    )
    collaboration_signals: dict[str, Any] = Field(
        default_factory=dict, description="Data on PRs, issues, and team participation"
    )
    raw_url: str = Field("", description="Link to the GitHub profile analyzed")


class KaggleSignals(BaseModel):
    """Heuristic data extracted from a user's Kaggle presence."""

    tier: str = Field("Novice", description="Kaggle progression tier")
    medals: dict[str, int] = Field(
        default_factory=dict, description="Counts for Gold, Silver, and Bronze medals"
    )
    ml_domains: list[str] = Field(
        default_factory=list, description="Domains identified from competition history"
    )
    notebook_quality_score: float = Field(
        ge=0, le=10, default=0, description="Average rating of public notebooks"
    )
    active_last_year: bool = Field(
        False, description="Whether there was activity in the last 365 days"
    )
    strongest_domain: str = Field(
        "", description="The field where the user has the most success"
    )


class SkillProfile(BaseModel):
    """Comprehensive snapshot of a user's technical capabilities and career context."""

    user_id: str = Field(..., description="Unique identifier of the user")
    target_role: str = Field(..., description="The role the user is aiming for")
    skills: list[SkillEntry] = Field(
        default_factory=list, description="List of verified and stated skills"
    )
    github_signals: Optional[GitHubSignals] = Field(
        None, description="Analyzed data from GitHub"
    )
    kaggle_signals: Optional[KaggleSignals] = Field(
        None, description="Analyzed data from Kaggle"
    )
    experience_years_by_domain: dict[str, float] = Field(
        default_factory=dict, description="Domain-specific years of experience"
    )
    certifications: list[str] = Field(
        default_factory=list, description="Formal certifications or badges earned"
    )
    stated_career_goal: Optional[str] = Field(
        None, description="The user's explicitly stated objective"
    )
    education_summary: str = Field(
        "", description="High-level summary of academic background"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the profile was first created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the most recent profile update",
    )

    def get_skill_level(self, skill_name: str) -> SkillLevel:
        """Checks the profile for a specific skill and returns its level."""
        normalized = skill_name.strip().lower()
        for s in self.skills:
            if s.name == normalized:
                return s.level
        return SkillLevel.NONE

    def top_skills(self, n: int = 10) -> list[SkillEntry]:
        """Returns the top N skills sorted by proficiency and assessment confidence."""
        return sorted(self.skills, key=lambda s: (s.level, s.confidence), reverse=True)[:n]