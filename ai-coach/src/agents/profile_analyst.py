"""CrewAI Agent: GitHub/Kaggle analysis."""

# src/agents/profile_analyst.py

from crewai import Agent, Task, Crew
from crewai_tools import tool
from src.tools.github_tools import analyze_github_profile
from src.tools.kaggle_tools import analyze_kaggle_notebooks
from src.state.schema import CoachState, UserProfile

profile_analyst = Agent(
    role="Expert Developer Profile Analyst",
    goal=(
        "Extract a precise, honest picture of the user's current technical skill level "
        "by analyzing their GitHub repositories, commit history, Kaggle notebooks, "
        "language usage patterns, and any uploaded documents. "
        "Identify both demonstrated strengths and clear skill gaps relative to the target role."
    ),
    backstory=(
        "You are a senior tech recruiter and engineering mentor with 15 years of experience "
        "evaluating developer portfolios at FAANG companies. You have an uncanny ability to "
        "distinguish between copy-paste code and genuine understanding. You are fair, thorough, "
        "and specific — you never say 'you need to improve Python' without citing which exact "
        "patterns are missing from their work."
    ),
    tools=[analyze_github_profile, analyze_kaggle_notebooks],
    llm="ollama/llama3.1:8b-instruct-q4_K_M",   # CrewAI v1.12 Ollama provider
    verbose=True,
    max_iter=3,
    allow_delegation=False,
)


def run_profile_analyst(state: CoachState) -> CoachState:
    """LangGraph node that invokes the Profile Analyst crew."""
    task = Task(
        description=(
            f"Analyze the profile for user '{state['user_profile'].name}'. "
            f"GitHub: {state['user_profile'].github_username or 'not provided'}. "
            f"Kaggle: {state['user_profile'].kaggle_username or 'not provided'}. "
            f"Target role: {state['user_profile'].target_role}. "
            "Return a structured JSON with: current_skills (list), skill_gaps (list), "
            "strengths (list), and recommended_starting_level (beginner/intermediate/advanced)."
        ),
        expected_output="JSON object with current_skills, skill_gaps, strengths, starting_level",
        agent=profile_analyst,
    )
    crew = Crew(agents=[profile_analyst], tasks=[task], verbose=False)
    result = crew.kickoff()

    # Parse CrewAI output and update state
    import json, re
    raw = result.raw
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        updated_profile = state["user_profile"].model_copy(update={
            "current_skills": parsed.get("current_skills", []),
            "skill_gaps":     parsed.get("skill_gaps", []),
        })
        return {**state, "user_profile": updated_profile, "is_new_user": False}
    return state