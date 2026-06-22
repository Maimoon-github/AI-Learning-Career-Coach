from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProjectSpec(BaseModel):
    """Schema for practice project specifications designed to test and build skills."""

    title: str = Field(
        ..., min_length=1, description="Concise, catchy name for the project"
    )
    description: str = Field(
        ..., min_length=10, description="Overview of the project goals and context"
    )
    problem_statement: str = Field(
        ...,
        min_length=10,
        description="The specific challenge or pain point the project addresses",
    )
    primary_skill: str = Field(
        ..., description="The main skill being exercised in this project"
    )
    secondary_skills: list[str] = Field(
        default_factory=list, description="Supporting skills used during implementation"
    )
    requirements: list[str] = Field(
        default_factory=list,
        min_length=1,
        description="Detailed functional requirements for the project",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        min_length=1,
        description="Conditions that must be met for the project to be considered complete",
    )
    tech_stack: dict[str, str] = Field(
        default_factory=dict,
        description="Recommended technologies and versions (e.g., {'fastapi': '0.100.0'})",
    )
    estimated_hours: float = Field(
        gt=0, le=80, description="Projected effort in hours (capped at 80 for modularity)"
    )
    artifact_type: Literal["API", "notebook", "CLI", "web-app", "model"] = Field(
        ..., description="The primary output format of the project"
    )
    difficulty: int = Field(
        ge=1, le=5, description="Complexity level (1: Beginner, 5: Expert)"
    )
    deliverables: list[str] = Field(
        default_factory=list,
        description="Specific files or artifacts to be produced (e.g., 'README.md', 'Dockerfile')",
    )
    anti_goals: list[str] = Field(
        default_factory=list,
        description="Explicitly out-of-scope items to prevent scope creep",
    )
    stretch_goals: list[str] = Field(
        default_factory=list,
        description="Optional features for higher proficiency levels",
    )
    suggested_file_structure: dict[str, Any] = Field(
        default_factory=dict,
        description="A recommended directory layout for the project",
    )
    calibration_notes: str = Field(
        "", description="Internal AI notes on how this project maps to the user's gaps"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of when this project specification was generated",
    )