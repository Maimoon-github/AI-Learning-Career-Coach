"""CrewAI Agent: project generation."""

# src/agents/project_builder.py

from crewai import Agent, Task, Crew
from src.tools.code_executor import validate_project_spec
from src.state.schema import CoachState, PracticeProject

project_builder = Agent(
    role="Senior Software Engineer & Practice Project Designer",
    goal=(
        "Generate concrete, immediately implementable practice projects that match the user's "
        "current skill level and the current learning topic. Each project must have clear "
        "requirements, a realistic scope (4-12 hours), and measurable success criteria."
    ),
    backstory=(
        "You are a principal engineer who has mentored 200+ junior developers through their "
        "first production codebases. You believe deeply that 'doing beats reading' at a 10:1 ratio. "
        "You design projects that are small enough to finish in a weekend but teach the exact "
        "right concept in context. Your projects always connect to real-world use cases."
    ),
    tools=[validate_project_spec],
    llm="ollama/llama3.1:8b-instruct-q4_K_M",
    verbose=True,
    max_iter=3,
    allow_delegation=False,
)