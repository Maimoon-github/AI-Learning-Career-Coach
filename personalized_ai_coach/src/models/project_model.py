from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProjectSpec(BaseModel):
    """Schema for practice project specifications."""
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=10)
    problem_statement: str = Field(..., min_length=10)
    primary_skill: str = Field(..., min_length=1)
    secondary_skills: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list, min_length=1)
    acceptance_criteria: list[str] = Field(default_factory=list, min_length=1)
    tech_stack: dict[str, str] = Field(default_factory=dict)  # component -> version
    estimated_hours: float = Field(gt=0, le=80)
    artifact_type: str = Field(..., pattern="^(API|notebook|CLI|web-app|model)$")
    difficulty: int = Field(ge=1, le=5)
    deliverables: list[str] = Field(default_factory=list)
    anti_goals: list[str] = Field(default_factory=list)
    stretch_goals: list[str] = Field(default_factory=list)
    suggested_file_structure: dict[str, Any] = Field(default_factory=dict)
    calibration_notes: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)