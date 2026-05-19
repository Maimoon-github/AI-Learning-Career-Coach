from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ResourceEntry(BaseModel):
    title: str
    url: str
    resource_type: str  # video | article | course | book | docs
    estimated_hours: float = Field(gt=0)
    is_free: bool = True
    quality_score: float = Field(ge=0, le=10)
    provider: str = ""


class WeekModule(BaseModel):
    week_number: int = Field(ge=1)
    primary_skill: str
    topics: list[str] = Field(min_length=1)
    estimated_hours: float = Field(gt=0)
    milestone: str
    review_topics: list[str] = Field(default_factory=list)
    resources: list[ResourceEntry] = Field(default_factory=list)
    is_review_week: bool = False

    @field_validator("primary_skill", "milestone")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class SkillGap(BaseModel):
    skill_name: str
    current_level: int = Field(ge=0, le=5)
    required_level: int = Field(ge=0, le=5)
    gap_severity: int = Field(ge=1, le=3)
    weeks_to_close: int = Field(ge=1)
    prerequisites: list[str] = Field(default_factory=list)
    learning_objective: str = ""
    priority_rank: int = 1

    @property
    def gap_size(self) -> int:
        return max(0, self.required_level - self.current_level)


class LearningPath(BaseModel):
    user_id: str
    target_role: str
    duration_weeks: int = Field(ge=1)
    hours_per_week: int = Field(ge=1)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    weeks: list[WeekModule] = Field(default_factory=list)
    total_hours: float = 0.0
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    validation_notes: list[str] = Field(default_factory=list)

    def current_week(self, week_offset: int = 0) -> WeekModule | None:
        idx = week_offset
        if 0 <= idx < len(self.weeks):
            return self.weeks[idx]
        return None

    def skills_in_scope(self) -> list[str]:
        return list({w.primary_skill for w in self.weeks})


class ProjectSpec(BaseModel):
    title: str
    description: str
    problem_statement: str
    primary_skill: str
    secondary_skills: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    tech_stack: dict[str, str] = Field(default_factory=dict)  # component -> version
    estimated_hours: float = Field(gt=0)
    artifact_type: str  # API | notebook | CLI | web-app | model
    difficulty: int = Field(ge=1, le=5)
    deliverables: list[str] = Field(default_factory=list)
    anti_goals: list[str] = Field(default_factory=list)
    stretch_goals: list[str] = Field(default_factory=list)
    suggested_file_structure: dict[str, Any] = Field(default_factory=dict)
    calibration_notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)